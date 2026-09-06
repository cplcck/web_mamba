#include "artifact.h"
#include "hash/hash.h"
extern "C" {
#include "hash/sha256/sha256.h"
}
#include <array>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
namespace capture {
std::string hash_text(const std::string & text) { return hash_sha256_hex(text.data(), text.size()); }
std::string hash_bytes(const std::vector<uint8_t> & bytes) { return hash_sha256_hex(bytes.data(), bytes.size()); }
std::string hash_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("identity.path: cannot read " + path.string());
    sha256_t context;
    sha256_init(&context);
    std::array<unsigned char, 65536> buffer;
    while (input.read(reinterpret_cast<char *>(buffer.data()), buffer.size()) || input.gcount())
        sha256_update(&context, buffer.data(), input.gcount());
    if (!input.eof()) throw std::runtime_error("identity.path: read failed " + path.string());
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&context, digest);
    std::ostringstream out;
    for (auto byte : digest) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return out.str();
}
json read_json(const fs::path & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("input.path: cannot read " + path.string());
    json value;
    input >> value;
    return value;
}
void write_json(const fs::path & path, const json & value) {
    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.open(path);
    output << value.dump(2) << '\n';
    output.close();
}
std::string command(const std::vector<std::string> & arguments) {
    // No shell expansion: model names and paths are data, not commands.
    int descriptors[2];
    if (pipe(descriptors)) throw std::runtime_error("provenance.pipe: failed");
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, descriptors[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, descriptors[0]);
    posix_spawn_file_actions_addclose(&actions, descriptors[1]);
    std::vector<char *> argv;
    for (const auto & arg : arguments) argv.push_back(const_cast<char *>(arg.c_str()));
    argv.push_back(nullptr);
    pid_t pid;
    const int result = posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(descriptors[1]);
    if (result) { close(descriptors[0]); throw std::runtime_error("provenance.command: spawn failed"); }
    std::string output;
    char buffer[4096];
    ssize_t count;
    while ((count = read(descriptors[0], buffer, sizeof(buffer))) > 0) output.append(buffer, count);
    close(descriptors[0]);
    int status;
    if (waitpid(pid, &status, 0) != pid || count < 0 || !WIFEXITED(status) || WEXITSTATUS(status))
        throw std::runtime_error("provenance.command: nonzero exit " + arguments[0]);
    return output;
}

Fixture parse_fixture(const json & input, int vocab) {
    if (input.at("schemaVersion") != 1 || input.at("kind") != "synthetic-token-ids")
        throw std::runtime_error("fixture.kind: unsupported format");
    for (const auto & field : {"sequenceId", "prefillLength", "outputsPerCall"})
        if (!input.at(field).is_number_integer()) throw std::runtime_error(std::string("fixture.") + field);
    if (input.at("sequenceId") != 0) throw std::runtime_error("fixture.sequenceId: must equal zero");
    if (input.at("prefillLength") != 16) throw std::runtime_error("fixture.prefillLength: must equal 16");
    if (input.at("outputsPerCall") != 1) throw std::runtime_error("fixture.outputsPerCall: must equal one");
    if (input.at("implicitSpecialTokens") != false) throw std::runtime_error("fixture.implicitSpecialTokens");
    if (!input.at("tokenIds").is_array() || input.at("tokenIds").size() != 17)
        throw std::runtime_error("fixture.tokenIds: exactly 17 IDs required");
    if (!input.at("positions").is_array() || input.at("positions").size() != 17)
        throw std::runtime_error("fixture.positions: exactly 17 positions required");
    Fixture result;
    for (size_t i = 0; i < 17; ++i) {
        const auto & token = input.at("tokenIds")[i];
        if (!token.is_number_integer() || token < 0 || token >= vocab)
            throw std::runtime_error("fixture.tokenIds[" + std::to_string(i) + "]: outside vocabulary");
        const auto & position = input.at("positions")[i];
        if (!position.is_number_integer() || position != i)
            throw std::runtime_error("fixture.positions[" + std::to_string(i) + "]: not aligned");
        result.ids.push_back(token.get<int32_t>());
        result.positions.push_back(position.get<int32_t>());
    }
    return result;
}

std::string call_kind(int previous, const std::vector<int32_t> & positions) {
    if (positions.empty()) throw std::runtime_error("call.positions: empty batch");
    for (size_t i = 0; i < positions.size(); ++i)
        if (positions[i] != previous + 1 + static_cast<int>(i))
            throw std::runtime_error("call.positions: not consecutive with retained state");
    return previous >= 0 && positions.size() == 1 ? "decode" : "prefill";
}
ArrayWriter::ArrayWriter(const fs::path & dir) : directory(dir) {
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.open(directory / "arrays.bin", std::ios::binary);
}
json ArrayWriter::append(const Array & array) {
    auto metadata = array.metadata;
    metadata["file"] = "arrays.bin";
    metadata["offsetBytes"] = offset;
    metadata["bytes"] = array.bytes.size();
    metadata["sha256"] = hash_bytes(array.bytes);
    metadata["byteOrder"] = "little";
    if (!array.bytes.empty()) stream.write(reinterpret_cast<const char *>(array.bytes.data()), array.bytes.size());
    offset += array.bytes.size();
    return metadata;
}
void ArrayWriter::close() { stream.close(); }
}
