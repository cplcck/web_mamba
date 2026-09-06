#pragma once
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <map>
#include <vector>
#include <string>

#include <exception>
#include <cstdint>
namespace capture {
using json = nlohmann::ordered_json;
struct Array {
    json metadata;
    std::vector<uint8_t> bytes;
};
std::vector<uint8_t> logical_copy(const ggml_tensor * tensor);
bool metadata_only(ggml_op op);
void validate_dependencies(const json & tensors);
struct Observer {
    bool active = false;
    std::map<ggml_tensor *, size_t> registry;
    std::vector<ggml_tensor *> asked;
    static bool callback(ggml_tensor * tensor, bool ask, void * user);
    std::string scenario = "test";
    uint64_t epoch = 0;
    json tensors = json::array();
    json events = json::array();
    std::vector<Array> arrays;
    std::map<ggml_backend_buffer_t, size_t> buffers;
    std::exception_ptr error;
    size_t register_tensor(ggml_tensor * tensor);
    json storage(const ggml_tensor * tensor);
    void retain(ggml_tensor * tensor, const std::string & phase);
    std::string id(size_t index) const;
};
}
