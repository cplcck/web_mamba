#pragma once
#include "artifact.h"
#include "llama.h"
namespace capture {
struct Snapshot {
    json metadata;
    std::vector<Array> arrays;
};
Snapshot snapshot(llama_context * context, const std::string & label, uint64_t epoch);
struct Run {
    json metadata;
    std::vector<std::vector<uint8_t>> comparable;
};
struct Schedule {
    Fixture fixture;
    std::string name;
    bool observer;
};
Run run(llama_model * model, const Schedule & schedule, ArrayWriter * writer);
void assert_graph(const Observer & observer);
}
