// Minimal reproducer: CPU plugin heap corruption / reset-regrow failure when
// ov::VariableState::set_state() SHRINKS the batch (outermost) dimension of a
// stateful model, then inference continues.
//
// Pure OpenVINO Runtime only (ov::Core / InferRequest / VariableState). No
// GenAI, no tokenizer, no audio, no Python. Deterministic constant inputs.
//
// Model: any stateful encoder-decoder *decoder* IR with dynamic-batch KV
// VariableStates and a `beam_idx` input (e.g. an optimum-exported Whisper
// decoder). See README.md for the export command.
//
// Usage: repro <decoder_model.xml> <mode> [N]
//   modes:
//     none    control: N rounds, prefill B=4 + infer B=4, NO set_state
//     same    control: N rounds, prefill B=4 + set_state SAME shape (B=4) + infer B=4
//     shrink  N rounds, prefill B=4 + set_state batch 4->3 + infer B=3   (Mode B)
//     modeA   prefill B=4 -> shrink 4->3 -> infer B=3 -> reset_state -> prefill B=4
//
// Only the batch-shrinking set_state (shrink / modeA) fails.

#include <openvino/openvino.hpp>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

static const size_t E = 1500;   // encoder sequence length (cross-KV state seq dim)
static const size_t P = 4;      // prefill prompt length
static const std::vector<int64_t> PROMPT = {1, 2, 3, 4};  // arbitrary valid token ids

static ov::Tensor enc_tensor(size_t B, size_t D) {
    ov::Tensor t(ov::element::f32, ov::Shape{B, E, D});
    std::fill_n(t.data<float>(), t.get_size(), 0.1f);
    return t;
}
static ov::Tensor i64(const std::vector<int64_t>& v, size_t r, size_t c) {
    ov::Tensor t(ov::element::i64, ov::Shape{r, c});
    std::memcpy(t.data<int64_t>(), v.data(), v.size() * sizeof(int64_t));
    return t;
}
static ov::Tensor beam(const std::vector<int32_t>& v) {
    ov::Tensor t(ov::element::i32, ov::Shape{v.size()});
    std::memcpy(t.data<int32_t>(), v.data(), v.size() * sizeof(int32_t));
    return t;
}

static void print_states(ov::InferRequest& req, const char* tag) {
    std::map<std::string, int> summary;  // "kind[dims]" -> count
    for (auto& s : req.query_state()) {
        auto sh = s.get_state().get_shape();
        bool cross = (sh.size() == 4 && sh[2] == E);   // cross-KV: seq dim == E
        std::string key = std::string(cross ? "cross" : "self") + "[";
        for (size_t i = 0; i < sh.size(); ++i) key += (i ? "," : "") + std::to_string(sh[i]);
        key += "]";
        summary[key]++;
    }
    std::cout << "    [" << tag << "] ";
    for (auto& kv : summary) std::cout << kv.first << " x" << kv.second << "  ";
    std::cout << std::endl;
}

// set_state on cross-KV states (seq dim == E). keep_rows rows are retained.
// Copy OUT into an independent tensor first (never retain the get_state view).
static void set_cross(ov::InferRequest& req, size_t keep_rows) {
    auto states = req.query_state();
    // one get_state per state, materialize fully, then set_state
    std::vector<std::pair<ov::VariableState, ov::Tensor>> work;
    for (auto& s : states) {
        ov::Tensor cur = s.get_state();
        ov::Tensor copy(cur.get_element_type(), cur.get_shape());
        cur.copy_to(copy);                       // independent buffer
        work.emplace_back(s, copy);
    }
    for (auto& [s, full] : work) {
        auto sh = full.get_shape();
        if (!(sh.size() == 4 && sh[2] == E)) continue;   // cross-KV only
        size_t keep = keep_rows ? keep_rows : sh[0];
        ov::Tensor out(full.get_element_type(), ov::Shape{keep, sh[1], sh[2], sh[3]});
        size_t row = sh[1] * sh[2] * sh[3];
        std::memcpy(out.data<float>(), full.data<float>(), keep * row * sizeof(float));
        s.set_state(out);
    }
}

static void prefill(ov::InferRequest& req, size_t B, size_t D) {
    std::vector<int64_t> ids(B * P);
    for (size_t i = 0; i < B; ++i) for (size_t c = 0; c < P; ++c) ids[i * P + c] = PROMPT[c];
    std::vector<int32_t> bi(B); std::iota(bi.begin(), bi.end(), 0);
    req.set_tensor("encoder_hidden_states", enc_tensor(B, D));
    req.set_tensor("input_ids", i64(ids, B, P));
    req.set_tensor("beam_idx", beam(bi));
    req.infer();
}
static void decode(ov::InferRequest& req, size_t W, size_t D) {
    std::vector<int64_t> tok(W, 100);
    std::vector<int32_t> bi(W); std::iota(bi.begin(), bi.end(), 0);
    req.set_tensor("encoder_hidden_states", enc_tensor(W, D));
    req.set_tensor("input_ids", i64(tok, W, 1));
    req.set_tensor("beam_idx", beam(bi));
    req.infer();
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: repro <decoder.xml> <none|same|shrink|modeA> [N]\n"; return 2; }
    std::string model_path = argv[1], mode = argv[2];
    int N = argc > 3 ? std::stoi(argv[3]) : 3;

    std::cout << "OpenVINO version: " << ov::get_openvino_version() << "\n";
    std::cout << "device: CPU\n";
    ov::Core core;
    auto model = core.read_model(model_path);
    size_t D = model->input("encoder_hidden_states").get_partial_shape()[2].get_length();
    auto compiled = core.compile_model(model, "CPU");
    std::cout << "mode: " << mode << "  N: " << N << "  D: " << D << "  E: " << E << "\n\n";

    if (mode == "modeA") {
        auto req = compiled.create_infer_request();
        prefill(req, 4, D);         print_states(req, "after prefill B=4");
        set_cross(req, 3);          print_states(req, "after set_state cross 4->3");
        decode(req, 3, D);          print_states(req, "after infer B=3");
        req.reset_state();          print_states(req, "after reset_state");
        std::cout << "  now regrow: prefill B=4 (expected to reinitialize state)\n";
        try {
            prefill(req, 4, D);
            std::cout << "  regrow prefill B=4: SUCCEEDED\n";
        } catch (const std::exception& e) {
            std::cout << "  regrow prefill B=4: FAILED -> " << e.what() << "\n";
        }
        return 0;
    }

    // control / Mode B: N independent fresh requests
    for (int round = 1; round <= N; ++round) {
        std::cout << "round " << round << ": " << std::flush;
        auto req = compiled.create_infer_request();
        prefill(req, 4, D);
        if (mode == "same")   set_cross(req, 4);   // set_state, SAME shape (no batch change)
        if (mode == "shrink") set_cross(req, 3);   // set_state, batch 4 -> 3
        size_t W = (mode == "shrink") ? 3 : 4;
        decode(req, W, D);
        std::cout << "ok" << std::endl;
        // req destroyed at end of scope
    }
    std::cout << "COMPLETED " << N << " rounds (mode=" << mode << ")\n";
    return 0;
}
