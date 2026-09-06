#include "observer.h"
#include <cmath>
#include <cstring>
#include <iterator>
#include <set>
#include <stdexcept>
namespace capture {

bool metadata_only(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
        op == GGML_OP_TRANSPOSE || op == GGML_OP_PERMUTE;
}

std::vector<uint8_t> logical_copy(const ggml_tensor * t) {
    const size_t block = ggml_blck_size(t->type), width = ggml_type_size(t->type);
    if (t->ne[0] % block != 0) throw std::runtime_error("tensor.ne[0]: partial dtype block");
    std::vector<uint8_t> out(ggml_nelements(t) / block * width);
    if (out.empty()) return out;
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, out.data(), 0, out.size());
    } else {
        std::vector<uint8_t> span(ggml_nbytes(t));
        ggml_backend_tensor_get(t, span.data(), 0, span.size());
        size_t dst = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
        for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
        for (int64_t i0 = 0; i0 < t->ne[0] / static_cast<int64_t>(block); ++i0) {
            const size_t src = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0*t->nb[0];
            std::memcpy(out.data() + dst, span.data() + src, width);
            dst += width;
        }
    }
    if (t->type == GGML_TYPE_F32) {
        for (size_t offset = 0; offset < out.size(); offset += sizeof(float)) {
            float value;
            std::memcpy(&value, out.data() + offset, sizeof(value));
            if (!std::isfinite(value)) throw std::runtime_error(std::string("tensor.values: nonfinite ") + t->name);
        }
    }
    return out;
}

std::string Observer::id(size_t index) const { return scenario + ":t" + std::to_string(index); }

json Observer::storage(const ggml_tensor * t) {
    auto buffer = t->buffer;
    if (!buffer && t->view_src) buffer = t->view_src->buffer;
    json s = {{"ggmlNbytes", ggml_nbytes(t)}, {"requiredAllocBytes", nullptr},
        {"bufferId", nullptr}, {"bufferOffsetBytes", nullptr}, {"bufferBytes", nullptr},
        {"observationEpoch", epoch}, {"allocatorSlotBytes", nullptr},
        {"allocatorSlotReason", "Pinned allocator does not expose per-tensor slot capacity"}};
    if (buffer && t->data) {
        const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
        const auto address = reinterpret_cast<uintptr_t>(t->data);
        const size_t capacity = ggml_backend_buffer_get_size(buffer);
        if (address < base || address - base > capacity || ggml_nbytes(t) > capacity - (address - base))
            throw std::runtime_error("storage.bufferOffsetBytes: range exceeds bufferBytes");
        auto entry = buffers.emplace(buffer, buffers.size()).first;
        s["bufferId"] = scenario + ":b" + std::to_string(entry->second);
        s["bufferOffsetBytes"] = address - base;
        s["bufferBytes"] = capacity;
        s["bufferType"] = ggml_backend_buffer_name(buffer);
        if (!t->view_src) s["requiredAllocBytes"] = ggml_backend_buffer_get_alloc_size(buffer, t);
    }
    return s;
}

size_t Observer::register_tensor(ggml_tensor * t) {
    auto found = registry.find(t);
    if (found != registry.end()) return found->second;
    const size_t index = registry.size();
    registry.emplace(t, index);
    tensors.push_back(nullptr); // Assign occurrence identity before descending; no pointer IDs.
    json src = json::array();
    for (auto * dependency : t->src)
        src.push_back(dependency ? json(id(register_tensor(dependency))) : json(nullptr));
    json view = t->view_src ? json(id(register_tensor(t->view_src))) : json(nullptr);
    const auto * root = t;
    size_t root_offset = 0;
    while (root->view_src) { root_offset += root->view_offs; root = root->view_src; }
    std::string role = "activation";
    if (std::string(t->name).rfind("cache_", 0) == 0) role = "state";
    else if (t->type == GGML_TYPE_I32) role = "index";
    else if (t->op == GGML_OP_NONE && t->buffer &&
             ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) role = "weight";
    tensors[index] = {{"id", id(index)}, {"name", t->name}, {"role", role},
        {"dtype", ggml_type_name(t->type)}, {"typeId", t->type},
        {"nativeShape", std::vector<int64_t>(t->ne, t->ne + 4)},
        {"strides", std::vector<size_t>(t->nb, t->nb + 4)},
        {"numel", ggml_nelements(t)}, {"typeBlockSize", ggml_blck_size(t->type)},
        {"typeSize", ggml_type_size(t->type)},
        {"logicalBytes", ggml_nelements(t) / ggml_blck_size(t->type) * ggml_type_size(t->type)},
        {"storage", storage(t)}, {"src", src}, {"viewSourceId", view},
        {"viewOffsetBytes", t->view_src ? json(t->view_offs) : json(nullptr)},
        {"viewRootId", id(registry.at(const_cast<ggml_tensor *>(root)))}, {"viewRootOffsetBytes", root_offset},
        {"op", ggml_op_name(t->op)}, {"opId", t->op},
        {"opParamsI32", std::vector<int32_t>(std::begin(t->op_params), std::end(t->op_params))},
        {"flags", t->flags}, {"classification", "observed"}, {"schedulerObserved", false},
        {"arithmeticExecution", false}, {"metadataOnly", metadata_only(t->op)}};
    // Leaf inputs have no callback. Copy the actual index values while still live.
    if (t->op == GGML_OP_NONE && t->type == GGML_TYPE_I32) retain(t, "dependency-input");
    return index;
}

void Observer::retain(ggml_tensor * t, const std::string & phase) {
    const auto index = registry.at(t);
    arrays.push_back({{{"tensorId", id(index)}, {"epoch", epoch}, {"phase", phase},
        {"dtype", ggml_type_name(t->type)}, {"shape", {t->ne[3], t->ne[2], t->ne[1], t->ne[0]}},
        {"axisLabels", {"ne3", "ne2", "ne1", "ne0"}}, {"order", "C"}}, logical_copy(t)});
}

bool Observer::callback(ggml_tensor * t, bool ask, void * user) {
    auto & self = *static_cast<Observer *>(user);
    if (!self.active) return false;
    try {
        if (self.error) return ask;
        ++self.epoch;
        if (ask) {
            const auto index = self.register_tensor(t);
            self.asked.push_back(t);
            auto & metadata = self.tensors[index];
            metadata["schedulerObserved"] = true;
            metadata["storage"] = self.storage(t);
            self.events.push_back({{"tensorId", self.id(index)}, {"epoch", self.epoch},
                {"phase", "ask"}, {"storage", metadata["storage"]}});
            return true;
        }
        const auto index = self.registry.at(t);
        const bool executed = !metadata_only(t->op) && ggml_nelements(t) > 0;
        self.tensors[index]["arithmeticExecution"] = executed;
        self.events.push_back({{"tensorId", self.id(index)}, {"epoch", self.epoch}, {"phase", "complete"},
            {"arithmeticExecution", executed}});
        if (executed) self.retain(t, "complete");
        return true;
    } catch (...) {
        self.error = std::current_exception(); // Rethrown immediately after decode; never export partial success.
        return ask;
    }
}

void validate_dependencies(const json & tensors) {
    std::set<std::string> ids;
    for (const auto & t : tensors) {
        if (!ids.insert(t.at("id").get<std::string>()).second)
            throw std::runtime_error("tensors.id: duplicate identity");
    }
    for (const auto & t : tensors) {
        for (const auto & dep : t.at("src"))
            if (!dep.is_null() && !ids.count(dep.get<std::string>()))
                throw std::runtime_error("tensors.src: missing tensor dependency");
        if (!t.at("viewSourceId").is_null() && !ids.count(t.at("viewSourceId").get<std::string>()))
            throw std::runtime_error("tensors.viewSourceId: missing view dependency");
    }
}
}
