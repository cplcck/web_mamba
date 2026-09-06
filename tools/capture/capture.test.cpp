#include "runtime.h"
#include "provenance.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>

static void require(bool ok, const char * invariant) {
    if (!ok) throw std::runtime_error(invariant);
}
static void rejects(const std::function<void()> & action, const std::string & field) {
    try { action(); }
    catch (const std::exception & error) {
        require(std::string(error.what()).find(field) != std::string::npos, "negative.named offending field");
        std::cout << "rejected " << field << '\n';
        return;
    }
    throw std::runtime_error("negative.accepted invalid " + field);
}
static void fixture_contract() {
    using namespace capture;
    json valid = {{"schemaVersion",1},{"kind","synthetic-token-ids"},{"sequenceId",0},
        {"prefillLength",16},{"outputsPerCall",1},{"implicitSpecialTokens",false},
        {"tokenIds",json::array()},{"positions",json::array()}};
    for (int i = 0; i < 17; ++i) { valid["tokenIds"].push_back(i+1); valid["positions"].push_back(i); }
    for (const auto & value : {json(-1),json(50280),json(1.5),json("1")}) {
        require(parse_fixture(valid,50280).ids.size() == 17, "control.fixture.tokenIds");
        auto invalid = valid; invalid["tokenIds"][3] = value;
        rejects([&] { parse_fixture(invalid,50280); }, "fixture.tokenIds[3]");
    }
    for (const auto & field : {"sequenceId","prefillLength","outputsPerCall","implicitSpecialTokens"}) {
        require(parse_fixture(valid,50280).positions.back() == 16, "control.fixture.fields");
        auto invalid = valid; invalid[field] = 8;
        rejects([&] { parse_fixture(invalid,50280); }, std::string("fixture.") + field);
    }
    require(parse_fixture(valid,50280).positions.back() == 16, "control.fixture.positions");
    auto invalid = valid; invalid["positions"][16] = 15;
    rejects([&] { parse_fixture(invalid,50280); }, "fixture.positions[16]");
    require(call_kind(15,{16}) == "decode", "control.call.retained_cache");
    rejects([] { call_kind(15,{17}); }, "call.positions");
    require(call_kind(-1,{0}) == "prefill", "empty_cache.single_token: not cached decode");
}
static void graph_contract() {
    // Given: actual graph arrays, including a disconnected cache mutation.
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 1);
    auto * leaf = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_set_name(leaf, "leaf"); ggml_set_input(leaf);
    auto * state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(state, "state"); ggml_set_input(state);
    auto * view = ggml_view_1d(ctx, leaf, 4, sizeof(float));
    auto * reshape = ggml_reshape_2d(ctx, view, 2, 2);
    auto * compute = ggml_add(ctx, reshape, reshape);
    auto * cpy = ggml_cpy(ctx, view, state);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, compute);
    ggml_build_forward_expand(graph, cpy);
    std::set<ggml_tensor *> expected(graph->nodes, graph->nodes + graph->n_nodes);
    expected.insert(graph->leafs, graph->leafs + graph->n_leafs);
    require(graph->n_nodes == 4 && graph->n_leafs == 2, "control.graph arrays");
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    const float input[] = {1,2,3,4,5,6,7,8};
    ggml_backend_buffer_clear(buffer, 0);
    ggml_backend_tensor_set(leaf, input, 0, sizeof(input));
    auto sched = ggml_backend_sched_new(&backend, nullptr, 1, 64, false, false);
    capture::Observer observer;
    observer.active = true;
    ggml_backend_sched_set_eval_callback(sched, capture::Observer::callback, &observer);
    // When: the real scheduler executes with the production observer callback.
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "control.compute status");
    if (observer.error) std::rethrow_exception(observer.error);
    // Then: callback registry exactly covers real nodes and leaves.
    std::set<ggml_tensor *> observed;
    for (const auto & entry : observer.registry) observed.insert(entry.first);
    require(observed == expected, "callback.registry: missing real node/leaf dependency");
    require(std::set<ggml_tensor *>(observer.asked.begin(), observer.asked.end()) ==
        std::set<ggml_tensor *>(graph->nodes, graph->nodes + graph->n_nodes), "callback.ask node completeness");
    float actual[4];
    ggml_backend_tensor_get(state, actual, 0, sizeof(actual));
    require(actual[0] == 2 && actual[3] == 5, "state.CPY: observer must continue evaluation");
    capture::validate_dependencies(observer.tensors);
    auto incomplete = observer.tensors;
    incomplete.erase(incomplete.begin() + observer.registry.at(leaf));
    rejects([&] { capture::validate_dependencies(incomplete); }, "tensors.src");
    capture::validate_dependencies(observer.tensors);
    auto broken_view = observer.tensors;
    broken_view[observer.registry.at(view)]["viewSourceId"] = "absent-view-root";
    rejects([&] { capture::validate_dependencies(broken_view); }, "tensors.viewSourceId");
    require(observer.tensors[observer.registry.at(view)]["arithmeticExecution"] == false, "view.metadata_only");
    require(observer.tensors[observer.registry.at(reshape)]["arithmeticExecution"] == false, "reshape.metadata_only");
    require(observer.tensors[observer.registry.at(cpy)]["viewSourceId"] == observer.id(observer.registry.at(state)), "CPY.destination alias");
    const auto copied_values = observer.arrays;
    require(copied_values.size() == 2, "compute and CPY values retained; views not materialized");
    ggml_backend_buffer_clear(buffer, 0);
    float retained[4];
    std::memcpy(retained,copied_values[0].bytes.data(),sizeof(retained));
    require(retained[0] == 4 && retained[3] == 10, "values.deep_copy before reused backing");
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
}
static void layout_contract() {
    auto * ctx = ggml_init({1024 * 1024,nullptr,true});
    auto backend = ggml_backend_cpu_init();
    auto * input = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,3,2);
    auto * transpose = ggml_transpose(ctx,input);
    auto * empty = ggml_view_1d(ctx,input,0,ggml_nbytes(input));
    auto * indices = ggml_new_tensor_1d(ctx,GGML_TYPE_I32,2);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx,backend);
    const float values[] = {1,2,3,4,5,6};
    ggml_backend_tensor_set(input,values,0,sizeof(values));
    const auto bytes = capture::logical_copy(transpose);
    const float expected[] = {1,4,2,5,3,6};
    require(bytes.size() == sizeof(expected) && std::memcmp(bytes.data(),expected,sizeof(expected)) == 0,
        "logical_copy.strides: transpose compact ne0-fastest order");
    const int32_t ids[] = {0,16};
    ggml_backend_tensor_set(indices,ids,0,sizeof(ids));
    capture::Observer observer;
    observer.epoch = 7;
    observer.register_tensor(indices);
    require(observer.arrays.size() == 1 && observer.arrays[0].bytes.size() == sizeof(ids) &&
        std::memcmp(observer.arrays[0].bytes.data(),ids,sizeof(ids)) == 0, "I32.leaf input values");
    const auto index = observer.register_tensor(empty);
    require(observer.tensors[index]["numel"] == 0 && observer.tensors[index]["storage"]["ggmlNbytes"] == 0 &&
        capture::logical_copy(empty).empty(), "empty_view.native zero payload/span preserved");
    require(observer.tensors[index]["storage"]["requiredAllocBytes"].is_null(), "empty_view.no independent allocation");
    for (float bad : {std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}) {
        ggml_backend_tensor_set(input,values,0,sizeof(values));
        require(capture::logical_copy(input).size() == sizeof(values), "control.finite values");
        ggml_backend_tensor_set(input,&bad,0,sizeof(bad));
        rejects([&] { capture::logical_copy(input); }, "tensor.values");
    }
    ggml_backend_buffer_free(buffer); ggml_backend_free(backend); ggml_free(ctx);
}
static void real_single_token(const char * manifest_path) {
    const auto manifest = capture::read_json(manifest_path);
    capture::provenance(manifest_path);
    llama_backend_init();
    auto params = llama_model_default_params(); params.n_gpu_layers = 0; params.use_extra_bufts = false;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(
        llama_model_load_from_file(manifest.at("ggufPath").get<std::string>().c_str(),params),llama_model_free);
    require(bool(model), "control.real_model");
    const auto single = capture::run(model.get(), {{{1},{0}},"single",true},nullptr);
    require(single.metadata.at("calls")[0].at("kind") == "prefill", "real.empty_cache.single_token not decode");
    require(single.metadata.at("calls")[0].at("previousPosition") == -1 &&
        single.metadata.at("calls")[0].at("position") == 0, "real.empty_cache positions");
    model.reset(); llama_backend_free();
}
int main(int argc, char ** argv) {
    try {
        for (const auto & key : {"GGML_CPU_DISABLE_FUSION","LLAMA_GRAPH_REUSE_DISABLE","OMP_NUM_THREADS",
                               "OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"})
            require(setenv(key,"1",1) == 0, "control.environment");
        fixture_contract();
        graph_contract();
        layout_contract();
        if (argc == 3 && std::string(argv[1]) == "--manifest") real_single_token(argv[2]);
        else require(argc == 1, "cli.test arguments");
        std::cout << "capture-contract: all invariants satisfied\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
