#include "provenance.h"
#include <sstream>
#include <stdexcept>
namespace capture {
static constexpr const char * source_commit = "8144f3192e5a3131cd043f284525e6ceebf82d0f";
json provenance(const fs::path & manifest_path) {
    const auto manifest = read_json(manifest_path);
    if (manifest.at("schemaVersion") != 1 || manifest.at("source").at("runtimeCommit") != source_commit ||
        manifest.at("source").at("path") != CAPTURE_LLAMA_SOURCE)
        throw std::runtime_error("manifest.source: incompatible build input");
    if (manifest.at("revision") != "1e76775f628fbf1350fbe4dbb3d971ba64af25a1" ||
        manifest.at("ggufSha256") != "af897eef0adfda25444be8fd2e6cbe3e858a107f7926625f7fb71d37f51bcaae")
        throw std::runtime_error("manifest.revision/ggufSha256: immutable input handoff mismatch");
    const auto & config = manifest.at("config");
    for (const auto & pair : std::vector<std::pair<std::string,int>>{{"num_hidden_layers",24},
             {"hidden_size",768},{"intermediate_size",1536},{"state_size",16},{"conv_kernel",4},{"time_step_rank",48}})
        if (config.at(pair.first) != pair.second) throw std::runtime_error("manifest.config." + pair.first);
    if (config.at("model_type") != "mamba" || manifest.at("outtype") != "f32")
        throw std::runtime_error("manifest.config.model_type/outtype");
    const fs::path snapshot = manifest.at("snapshotPath").get<std::string>();
    if (read_json(snapshot / "config.json") != config) throw std::runtime_error("snapshot.config: changed");
    for (const auto & payload : manifest.at("payloads").items()) {
        if (hash_file(snapshot / payload.key()) != payload.value().at("sha256"))
            throw std::runtime_error("snapshot.payloads." + payload.key() + ": hash mismatch");
    }
    if (hash_file(manifest.at("ggufPath").get<std::string>()) != manifest.at("ggufSha256"))
        throw std::runtime_error("manifest.ggufSha256: payload hash mismatch");
    for (const auto & fixture : manifest.at("fixtures").items())
        if (hash_file(fixture.value().at("path").get<std::string>()) != fixture.value().at("sha256"))
            throw std::runtime_error("manifest.fixtures." + fixture.key() + ": hash mismatch");
    const auto git = [](std::vector<std::string> args) {
        std::vector<std::string> argv = {"timeout", "30", "git", "--no-optional-locks", "-C", CAPTURE_LLAMA_SOURCE};
        argv.insert(argv.end(), args.begin(), args.end());
        return command(argv);
    };
    if (git({"rev-parse","HEAD"}) != std::string(source_commit) + "\n")
        throw std::runtime_error("source.commit: stale runtime baseline");
    const auto diff = git({"diff","HEAD","--binary","--",".",":(exclude).pi",":(exclude).omo"});
    if (hash_text(diff) != manifest.at("source").at("productPatchSha256"))
        throw std::runtime_error("source.productPatchSha256: modified product source");
    const auto status = git({"status","--porcelain","--untracked-files=all"});
    std::istringstream lines(status);
    std::string line;
    while (std::getline(lines, line))
        if (line != " D .pi/gg/SYSTEM.md" && line.rfind("?? .omo/",0) != 0)
            throw std::runtime_error("source.status: out-of-baseline product change " + line);
    auto build = read_json(fs::path(CAPTURE_BUILD_DIR) / "build-identity.json");
    for (const auto & file : build.at("files").items())
        if (hash_file(file.key()) != file.value()) throw std::runtime_error("build.files: stale " + file.key());
    if (hash_file(build.at("compiler").get<std::string>()) != build.at("compilerSha256"))
        throw std::runtime_error("build.compilerSha256: stale compiler");
    std::ifstream cache(fs::path(CAPTURE_BUILD_DIR) / "CMakeCache.txt");
    json flags = json::object();
    while (std::getline(cache,line)) {
        const auto colon = line.find(':'), equals = line.find('=');
        if (colon != std::string::npos && equals > colon && equals != std::string::npos &&
            (line.rfind("GGML_",0) == 0 || line.rfind("LLAMA_",0) == 0 || line.rfind("CMAKE_",0) == 0 ||
             line.rfind("BUILD_SHARED_LIBS:",0) == 0))
            flags[line.substr(0,colon)] = line.substr(equals+1);
    }
    if (!cache.eof()) throw std::runtime_error("build.CMakeCache: read failed");
    build["cmakeFlags"] = flags;
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string cpu;
    while (std::getline(cpuinfo,line)) if (line.rfind("model name",0) == 0) {
        const auto start = line.find(':');
        cpu = line.substr(line.find_first_not_of(" \t",start+1));
        break;
    }
    if (cpu.empty()) throw std::runtime_error("environment.cpu: not observed");
    return {{"preparationCaptureId",manifest.at("captureId")},{"manifestPath",fs::absolute(manifest_path).string()},
        {"manifestSha256",hash_file(manifest_path)},{"ggufSha256",manifest.at("ggufSha256")},
        {"revision",manifest.at("revision")},{"snapshotPath",snapshot.string()},
        {"source",{{"commit",source_commit},{"productPatchSha256",hash_text(diff)},
            {"status",status},{"worktreeDiffSha256",hash_text(git({"diff","HEAD","--binary"}))}}},
        {"build",build},{"environment",{{"GGML_CPU_DISABLE_FUSION","1"},{"LLAMA_GRAPH_REUSE_DISABLE","1"},
            {"OMP_NUM_THREADS","1"},{"OPENBLAS_NUM_THREADS","1"},{"MKL_NUM_THREADS","1"}}},
        {"cpu",cpu},{"platform",command({"timeout","10","uname","-a"})},{"publicValidationStatus","unvalidated"}};
}
Fixture supplied_fixture(const fs::path & path, const std::string & selected, const json & manifest) {
    const auto input = read_json(path);
    const auto canonical = read_json(manifest.at("fixtures").at("token-ids.json").at("path").get<std::string>());
    const auto calibration = read_json(manifest.at("fixtures").at("calibration-ids.json").at("path").get<std::string>());
    auto resolved = input;
    if (input.contains("fixtures")) {
        if (selected.empty() || input != calibration) throw std::runtime_error("fixture.case: declared calibration case required");
        resolved = canonical;
        bool found = false;
        for (const auto & fixture : input.at("fixtures")) if (fixture.at("id") == selected) {
            resolved["tokenIds"] = fixture.at("tokenIds"); found = true;
        }
        if (!found) throw std::runtime_error("fixture.case: unknown calibration case");
    } else if (!selected.empty()) throw std::runtime_error("fixture.case: only valid with calibration collection");
    const auto fixture = parse_fixture(resolved, manifest.at("config").at("vocab_size"));
    bool declared = resolved.at("tokenIds") == canonical.at("tokenIds");
    for (const auto & alternative : calibration.at("fixtures"))
        declared = declared || resolved.at("tokenIds") == alternative.at("tokenIds");
    if (!declared) throw std::runtime_error("fixture.tokenIds: not a predeclared canonical/calibration case");
    return fixture;
}
}
