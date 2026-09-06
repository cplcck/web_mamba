#include "runtime.h"
#include "llama-context.h"
#include "llama-memory-recurrent.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
namespace capture {
static llama_memory_recurrent & recurrent(llama_context * context) {
    auto * memory = dynamic_cast<llama_memory_recurrent *>(llama_get_memory(context));
    if (!memory) throw std::runtime_error("memory.kind: expected llama_memory_recurrent");
    if (memory->n_rs_seq != 0 || memory->rs_idx.size() != 1)
        throw std::runtime_error("memory.n_rs_seq/n_seq_max: unsupported recurrent configuration");
    return *memory;
}

Snapshot snapshot(llama_context * context, const std::string & label, uint64_t epoch) {
    llama_synchronize(context);
    const auto & memory = recurrent(context);
    Snapshot result;
    result.metadata = {{"label", label}, {"epoch", epoch}, {"head", memory.head},
        {"size", memory.size}, {"used", memory.used}, {"n", memory.n}, {"rs_z", memory.rs_z},
        {"rs_idx", memory.rs_idx}, {"n_rs_seq", memory.n_rs_seq},
        {"position", llama_memory_seq_pos_max(llama_get_memory(context), 0)},
        {"cells", json::array()}, {"activeCell", nullptr}, {"activeRow", nullptr}};
    int row = -1;
    for (size_t i = 0; i < memory.cells.size(); ++i) {
        const auto & cell = memory.cells[i];
        result.metadata["cells"].push_back({{"index", i}, {"pos", cell.pos},
            {"src", cell.src}, {"src0", cell.src0}, {"tail", cell.tail}, {"sequenceIds", cell.seq_id}});
        if (cell.has_seq_id(0)) {
            if (row >= 0) throw std::runtime_error("memory.activeCell: multiple sequence0 cells");
            // Exact state_write selection, rollback disabled. Do not assume head is source.
            row = cell.src >= 0 ? cell.src : static_cast<int>(i);
            if (row < 0 || static_cast<uint32_t>(row) >= memory.size)
                throw std::runtime_error("memory.activeRow: invalid cell.src");
            result.metadata["activeCell"] = i;
            result.metadata["activeRow"] = row;
        }
    }
    if (memory.r_l.size() != 24 || memory.s_l.size() != 24)
        throw std::runtime_error("memory.layers: expected 24");
    for (size_t layer = 0; layer < 24; ++layer) {
        for (const auto & family : {std::string("R"), std::string("S")}) {
            const auto * tensor = family == "R" ? memory.r_l[layer] : memory.s_l[layer];
            const int width = family == "R" ? 3 : 16;
            if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] != 1536*width ||
                tensor->ne[1] != memory.size || !ggml_is_contiguous(tensor))
                throw std::runtime_error("memory.tensor: unexpected native dtype/layout");
            auto bytes = logical_copy(tensor);
            json shape = {memory.size, 1536, width};
            if (row >= 0) {
                const auto begin = bytes.begin() + row * tensor->nb[1];
                bytes = std::vector<uint8_t>(begin, begin + tensor->nb[1]);
                shape = {1, 1536, width};
            } else if (std::any_of(bytes.begin(), bytes.end(), [](uint8_t byte) { return byte != 0; })) {
                throw std::runtime_error("memory.initial: full buffers must be bytewise zero");
            }
            result.arrays.push_back({{{"kind", "state"}, {"family", family}, {"layer", layer},
                {"snapshot", label}, {"epoch", epoch}, {"dtype", "f32"}, {"shape", shape},
                {"axisLabels", {row >= 0 ? "sequence" : "cell", "inner", family == "R" ? "history" : "state"}},
                {"order", "C"}, {"nativeShape", {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]}},
                {"nativeStrides", {tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]}},
                {"sourceRow", row >= 0 ? json(row) : json(nullptr)}, {"fullInitialBuffer", row < 0}}, std::move(bytes)});
        }
    }
    return result;
}

void assert_graph(const Observer & observer) {
    validate_dependencies(observer.tensors);
    if (observer.asked.empty()) throw std::runtime_error("graph.callbacks: absent");
    std::map<std::string, const json *> by_id;
    for (const auto & tensor : observer.tensors) by_id.emplace(tensor.at("id"), &tensor);
    std::set<int> conv_layers, scan_layers, r_writes, s_writes;
    for (const auto & tensor : observer.tensors) {
        if (!tensor.at("schedulerObserved").get<bool>()) continue;
        const std::string op = tensor.at("op");
        if (op == "SSM_CONV") {
            const std::string weight = by_id.at(tensor.at("src")[1])->at("name");
            int layer = -1;
            if (std::sscanf(weight.c_str(), "blk.%d.ssm_conv1d.weight", &layer) != 1 ||
                weight != "blk." + std::to_string(layer) + ".ssm_conv1d.weight" || !conv_layers.insert(layer).second)
                throw std::runtime_error("graph.SSM_CONV: invalid/duplicate layer weight");
        }
        if (op == "SSM_SCAN") {
            const std::string weight = by_id.at(tensor.at("src")[3])->at("name");
            int layer = -1;
            if (std::sscanf(weight.c_str(), "blk.%d.ssm_a", &layer) != 1 ||
                weight != "blk." + std::to_string(layer) + ".ssm_a" || !scan_layers.insert(layer).second)
                throw std::runtime_error("graph.SSM_SCAN: invalid/duplicate layer weight");
        }
        if (op == "CPY" && tensor.at("numel") > 0) {
            const auto & root = *by_id.at(tensor.at("viewRootId"));
            const std::string name = root.at("name");
            for (int layer = 0; layer < 24; ++layer) {
                if (name == "cache_r_l" + std::to_string(layer)) r_writes.insert(layer);
                if (name == "cache_s_l" + std::to_string(layer)) s_writes.insert(layer);
            }
        }
    }
    for (int layer = 0; layer < 24; ++layer)
        if (!conv_layers.count(layer) || !scan_layers.count(layer) || !r_writes.count(layer) || !s_writes.count(layer))
            throw std::runtime_error("graph.layer[" + std::to_string(layer) + "]: missing conv/scan/R/S writeback");
}

Run run(llama_model * model, const Schedule & schedule, ArrayWriter * writer) {
    Observer observer;
    auto params = llama_context_default_params();
    params.n_ctx = 32; params.n_batch = 32; params.n_ubatch = 32;
    params.n_seq_max = 1; params.n_rs_seq = 0;
    params.n_outputs_max = 1; params.n_outputs_max_per_seq = 1;
    params.n_threads = 1; params.n_threads_batch = 1;
    params.offload_kqv = false; params.op_offload = false;
    params.type_k = GGML_TYPE_F32; params.type_v = GGML_TYPE_F32;
    params.no_perf = true; params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.cb_eval = schedule.observer ? Observer::callback : nullptr;
    params.cb_eval_user_data = &observer;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model, params), llama_free);
    if (!context) throw std::runtime_error("context.create: failed");
    const auto & effective = context->get_cparams();
    if (effective.n_ctx < 32 || effective.n_batch < 32 || effective.n_ubatch < 32 ||
        effective.n_threads != 1 || effective.n_threads_batch != 1 || effective.n_seq_max != 1 ||
        effective.n_rs_seq != 0 || effective.offload_kqv || effective.op_offload || effective.warmup)
        throw std::runtime_error("context.effective: CPU capture configuration differs");
    auto sched = context->get_sched();
    json backends = json::array();
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        auto backend = ggml_backend_sched_get_backend(sched, i);
        if (!ggml_backend_is_cpu(backend)) throw std::runtime_error("context.backend: CPU only");
        backends.push_back(ggml_backend_name(backend));
    }
    Run result;
    result.metadata = {{"schedule", schedule.name}, {"observerEnabled", schedule.observer},
        {"configured", {{"n_ctx",32},{"n_batch",32},{"n_ubatch",32},{"n_seq_max",1},{"n_rs_seq",0},
            {"n_threads",1},{"n_threads_batch",1},{"offload_kqv",false},{"op_offload",false},
            {"n_outputs_max",1},{"n_outputs_max_per_seq",1},{"type_k","f32"},{"type_v","f32"}}},
        {"effective", {{"n_ctx",effective.n_ctx},{"n_batch",effective.n_batch},{"n_ubatch",effective.n_ubatch},
            {"n_seq_max",effective.n_seq_max},{"n_rs_seq",effective.n_rs_seq},{"n_threads",effective.n_threads},
            {"n_threads_batch",effective.n_threads_batch},{"offload_kqv",effective.offload_kqv},
            {"op_offload",effective.op_offload},{"warmup",effective.warmup},{"backends",backends}}},
        {"snapshots", json::array()}, {"calls", json::array()}};
    uint64_t epoch = 1;
    auto store_snapshot = [&](Snapshot snap) {
        snap.metadata["arrays"] = json::array();
        for (const auto & array : snap.arrays) {
            result.comparable.push_back(array.bytes);
            if (writer) snap.metadata["arrays"].push_back(writer->append(array));
        }
        result.metadata["snapshots"].push_back(snap.metadata);
        return snap;
    };
    store_snapshot(snapshot(context.get(), "before-prefill", epoch++));
    std::vector<std::pair<int,int>> calls = schedule.name == "split" ?
        std::vector<std::pair<int,int>>{{0,16},{16,17}} : std::vector<std::pair<int,int>>{{0,static_cast<int>(schedule.fixture.ids.size())}};
    Snapshot after_prefill;
    for (const auto & range : calls) {
        const int begin = range.first, end = range.second;
        const std::vector<int32_t> positions(schedule.fixture.positions.begin()+begin, schedule.fixture.positions.begin()+end);
        const int previous = llama_memory_seq_pos_max(llama_get_memory(context.get()), 0);
        const auto kind = call_kind(previous, positions);
        if (begin) {
            auto before = snapshot(context.get(), "before-decode", epoch++);
            for (size_t i = 0; i < before.arrays.size(); ++i)
                if (before.arrays[i].bytes != after_prefill.arrays[i].bytes)
                    throw std::runtime_error("state.continuity: after-prefill differs before-decode");
            if (before.metadata.at("cells") != after_prefill.metadata.at("cells"))
                throw std::runtime_error("state.continuity: cell mapping changed before-decode");
            store_snapshot(std::move(before));
            result.metadata["prefillDecodeContinuityBytewise"] = true;
        }
        observer = Observer{};
        observer.scenario = schedule.name + ":" + kind;
        observer.epoch = epoch;
        observer.active = true;
        auto batch = llama_batch_init(end-begin, 0, 1);
        batch.n_tokens = end-begin;
        for (int i = begin; i < end; ++i) {
            const int j = i-begin;
            batch.token[j] = schedule.fixture.ids[i]; batch.pos[j] = schedule.fixture.positions[i];
            batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0; batch.logits[j] = i == end-1;
        }
        const int status = llama_decode(context.get(), batch);
        llama_batch_free(batch);
        observer.active = false;
        llama_synchronize(context.get());
        if (observer.error) std::rethrow_exception(observer.error);
        if (status) throw std::runtime_error("llama_decode.status: " + std::to_string(status));
        epoch = observer.epoch + 1;
        if (schedule.observer) assert_graph(observer);
        const int position = llama_memory_seq_pos_max(llama_get_memory(context.get()), 0);
        if (position != end-1) throw std::runtime_error("memory.position: not final input index");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * logits = llama_get_logits_ith(context.get(), end-begin-1);
        if (!logits) throw std::runtime_error("logits.row: missing final output");
        Array output{{{"kind","logits"},{"dtype","f32"},{"shape",{vocab}},{"axisLabels",{"vocabulary"}},
            {"order","C"},{"epoch",epoch++},{"inputIndex",end-1},{"tokenId",schedule.fixture.ids[end-1]},
            {"sequenceId",0},{"position",position},{"outputRow",0},{"batchTokenIndex",end-begin-1},
            {"vocabStart",0},{"vocabEndExclusive",vocab}},
            std::vector<uint8_t>(vocab*sizeof(float))};
        for (int i = 0; i < vocab; ++i) if (!std::isfinite(logits[i])) throw std::runtime_error("logits.values: nonfinite");
        std::memcpy(output.bytes.data(), logits, output.bytes.size());
        result.comparable.push_back(output.bytes);
        json call = {{"kind",kind},{"positions",positions},{"sequenceId",0},{"previousPosition",previous},
            {"position",position},{"inputBegin",begin},{"inputEndExclusive",end},{"O",1},
            {"P",1},{"T",end-begin},{"Q",end-begin},{"decodeExit",status}};
        if (writer) {
            call["logits"] = writer->append(output);
            call["tensors"] = observer.tensors;
            call["events"] = observer.events;
            call["arrays"] = json::array();
            for (const auto & array : observer.arrays) call["arrays"].push_back(writer->append(array));
        }
        result.metadata["calls"].push_back(std::move(call));
        after_prefill = store_snapshot(snapshot(context.get(), "after-" + kind, epoch++));
    }
    return result;
}
}
