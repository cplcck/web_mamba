#include "runtime.h"
#include "provenance.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <cstdlib>
int main(int argc, char ** argv) {
    using namespace capture;
    try {
        std::map<std::string,std::string> args;
        for (int i = 1; i < argc; i += 2) {
            const std::string key = argv[i];
            if (i+1 == argc || (key != "--manifest" && key != "--tokens" && key != "--schedule" &&
                key != "--out" && key != "--case") || !args.emplace(key,argv[i+1]).second)
                throw std::runtime_error("cli.arguments: expected --manifest --tokens --schedule split|fresh --out [--case ID]");
        }
        for (const auto & key : {"--manifest","--tokens","--schedule","--out"})
            if (!args.count(key)) throw std::runtime_error(std::string("cli.arguments: missing ") + key);
        if (args.at("--schedule") != "split" && args.at("--schedule") != "fresh")
            throw std::runtime_error("cli.schedule: expected split or fresh");
        const fs::path out = fs::absolute(args.at("--out"));
        if (fs::exists(out)) throw std::runtime_error("output.path: already exists; never overwrite or resume partial captures");
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian) != 1) throw std::runtime_error("environment.byteOrder: little-endian required");
        for (const auto & key : {"GGML_CPU_DISABLE_FUSION","LLAMA_GRAPH_REUSE_DISABLE","OMP_NUM_THREADS",
                               "OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"})
            if (setenv(key,"1",1)) throw std::runtime_error(std::string("environment.") + key);
        const auto manifest = read_json(args.at("--manifest"));
        const auto fixture = supplied_fixture(args.at("--tokens"),args.count("--case") ? args.at("--case") : "",manifest);
        auto identity = provenance(args.at("--manifest"));
        identity["tokensPath"] = fs::absolute(args.at("--tokens")).string();
        identity["tokensSha256"] = hash_file(args.at("--tokens"));
        identity["selectedCase"] = args.count("--case") ? json(args.at("--case")) : json(nullptr);
        identity["tokenIds"] = fixture.ids;
        identity["positions"] = fixture.positions;
        identity["schedule"] = args.at("--schedule");
        identity["scheduleCalls"] = args.at("--schedule") == "split" ? json{{0,16},{16,17}} : json{{0,17}};
        identity["scheduleSha256"] = hash_text(identity.at("scheduleCalls").dump());
        identity["command"] = std::vector<std::string>(argv,argv+argc);
        llama_backend_init();
        struct BackendGuard { ~BackendGuard() { llama_backend_free(); } } backend_guard;
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0; params.use_extra_bufts = false; params.no_host = false;
        params.check_tensors = true;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(
            llama_model_load_from_file(manifest.at("ggufPath").get<std::string>().c_str(),params),llama_model_free);
        if (!model) throw std::runtime_error("model.load: failed");
        if (llama_model_n_layer(model.get()) != 24 || llama_model_n_embd(model.get()) != 768 ||
            llama_vocab_n_tokens(llama_model_get_vocab(model.get())) != manifest.at("config").at("vocab_size"))
            throw std::runtime_error("model.dimensions/vocabulary: incompatible GGUF");
        identity["modelConfigured"] = {{"n_gpu_layers",0},{"use_extra_bufts",false},{"no_host",false},{"check_tensors",true}};
        identity["systemInfo"] = llama_print_system_info();
        fs::create_directories(out.parent_path());
        if (!fs::create_directory(out)) throw std::runtime_error("output.path: concurrent creation");
        write_json(out / "incomplete.json", {{"status","incomplete"},{"identity",identity}});
        ArrayWriter writer(out);
        auto observed = run(model.get(), {fixture,args.at("--schedule"),true}, &writer);
        writer.close();
        const auto control = run(model.get(), {fixture,args.at("--schedule"),false}, nullptr);
        if (observed.comparable != control.comparable)
            throw std::runtime_error("observer.noninterference: logits/state bytes differ from observer-off control");
        observed.metadata["observerNoninterference"] = {{"comparison","bytewise"},{"equal",true},
            {"arrayCount",observed.comparable.size()},{"controlSettings",control.metadata.at("effective")}};
        auto & pairs = observed.metadata["observerNoninterference"]["pairs"] = json::array();
        for (size_t i = 0; i < observed.comparable.size(); ++i)
            pairs.push_back({{"index",i},{"observedSha256",hash_bytes(observed.comparable[i])},
                {"controlSha256",hash_bytes(control.comparable[i])}});
        identity["arraysSha256"] = hash_file(out / "arrays.bin");
        identity["runMetadataSha256"] = hash_text(observed.metadata.dump());
        const std::string capture_id = "llama-" + hash_text(identity.dump());
        json document = {{"schemaVersion",1},{"rawFormat","mamba-native-capture-v1"},{"captureId",capture_id},
            {"ggufSha256",manifest.at("ggufSha256")},{"status","captured-unvalidated"},{"identity",identity},
            {"run",observed.metadata}};
        write_json(out / "capture.json",document);
        write_json(out / "complete.json",{{"status","captured-unvalidated"},{"captureId",capture_id},
            {"captureSha256",hash_file(out / "capture.json")},{"arraysSha256",identity.at("arraysSha256")}});
        fs::remove(out / "incomplete.json");
        std::cout << json{{"status","captured-unvalidated"},{"out",out.string()},{"captureId",capture_id}}.dump() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << capture::json{{"status","error"},{"error",error.what()}}.dump() << '\n';
        return 1;
    }
}
