// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"

namespace ov::genai::silero_vad {

constexpr const char* SEQUENCE_MODEL_FILE = "silero_vad_16k_sequence.onnx";
constexpr size_t CONTEXT_SAMPLES = 64;
constexpr size_t STATE_SIZE = 128;
constexpr size_t MAX_BLOCK_FRAMES = 512;

/**
 * @brief Silero VAD 16 kHz sequence model producing one speech probability per 512-sample window.
 */
class SileroVADModel {
public:
    SileroVADModel(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties);

    /**
     * @brief Runs the model over the whole audio with fresh recurrent state.
     * @return One probability per window; the last window is zero-padded. Empty for empty audio.
     */
    std::vector<float> get_speech_probabilities(const std::vector<float>& audio);

private:
    ov::InferRequest m_request;
};

}  // namespace ov::genai::silero_vad
