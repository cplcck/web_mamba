#pragma once
#include "observer.h"
#include <filesystem>
#include <fstream>
namespace capture {
namespace fs = std::filesystem;
std::string hash_file(const fs::path & path);
std::string hash_bytes(const std::vector<uint8_t> & bytes);
std::string hash_text(const std::string & text);
json read_json(const fs::path & path);
void write_json(const fs::path & path, const json & value);
std::string command(const std::vector<std::string> & arguments);
struct Fixture {
    std::vector<int32_t> ids;
    std::vector<int32_t> positions;
};
Fixture parse_fixture(const json & input, int vocab);
std::string call_kind(int previous_position, const std::vector<int32_t> & positions);
struct ArrayWriter {
    fs::path directory;
    std::ofstream stream;
    uint64_t offset = 0;
    explicit ArrayWriter(const fs::path & directory);
    json append(const Array & array);
    void close();
};
}
