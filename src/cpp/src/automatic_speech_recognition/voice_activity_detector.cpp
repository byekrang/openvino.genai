// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/automatic_speech_recognition/voice_activity_detector.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>

#include "automatic_speech_recognition/vad/silero_vad_model.hpp"
#include "automatic_speech_recognition/vad/silero_vad_segmenter.hpp"
#include "openvino/core/except.hpp"

namespace {

ov::genai::SileroVADConfig extract_config(ov::AnyMap& properties) {
    ov::genai::SileroVADConfig config;
    const auto it = properties.find(ov::genai::silero_vad_config.name());
    if (it != properties.end()) {
        config = it->second.as<ov::genai::SileroVADConfig>();
        properties.erase(it);
    }
    config.validate();
    return config;
}

}  // namespace

namespace ov::genai {

class VoiceActivityDetector::Impl {
public:
    Impl(const std::filesystem::path& models_path, const std::string& device, ov::AnyMap properties)
        : m_config(extract_config(properties)),
          m_model(models_path, device, properties) {}

    std::vector<SpeechSegment> detect(const std::vector<float>& audio) {
        if (audio.empty()) {
            return {};
        }
        const auto non_finite = std::find_if(audio.begin(), audio.end(), [](float sample) {
            return !std::isfinite(sample);
        });
        OPENVINO_ASSERT(non_finite == audio.end(),
                        "Audio contains a non-finite sample ",
                        *non_finite,
                        " at index ",
                        std::distance(audio.begin(), non_finite));
        const std::vector<float> probabilities = m_model.get_speech_probabilities(audio);
        for (size_t index = 0; index < probabilities.size(); ++index) {
            OPENVINO_ASSERT(std::isfinite(probabilities[index]),
                            "Silero VAD model produced a non-finite speech probability ",
                            probabilities[index],
                            " for window ",
                            index);
        }
        return silero_vad::get_speech_segments(probabilities, audio.size(), m_config);
    }

private:
    SileroVADConfig m_config;
    silero_vad::SileroVADModel m_model;
};

VoiceActivityDetector::VoiceActivityDetector(const std::filesystem::path& models_path,
                                             const std::string& device,
                                             const ov::AnyMap& properties)
    : m_impl(std::make_unique<Impl>(models_path, device, properties)) {}

VoiceActivityDetector::~VoiceActivityDetector() = default;

std::vector<SpeechSegment> VoiceActivityDetector::detect(const std::vector<float>& audio) {
    return m_impl->detect(audio);
}

void SileroVADConfig::validate() const {
    OPENVINO_ASSERT(std::isfinite(threshold) && threshold > 0.0 && threshold < 1.0,
                    "SileroVADConfig.threshold must be in (0, 1), got ",
                    threshold);
    if (neg_threshold.has_value()) {
        const double value = *neg_threshold;
        OPENVINO_ASSERT(std::isfinite(value) && value > 0.0 && value <= threshold,
                        "SileroVADConfig.neg_threshold must be in (0, threshold], got ",
                        value,
                        " with threshold ",
                        threshold);
    }
    OPENVINO_ASSERT(!std::isnan(max_speech_duration_s) && max_speech_duration_s > 0.0,
                    "SileroVADConfig.max_speech_duration_s must be positive or infinity, got ",
                    max_speech_duration_s);
    for (const auto& [name, value] : {std::pair<const char*, size_t>{"min_speech_duration_ms", min_speech_duration_ms},
                                      {"min_silence_duration_ms", min_silence_duration_ms},
                                      {"speech_pad_ms", speech_pad_ms},
                                      {"min_silence_at_max_speech_ms", min_silence_at_max_speech_ms}}) {
        OPENVINO_ASSERT(value <= silero_vad::MAX_DURATION_MS,
                        "SileroVADConfig.",
                        name,
                        " must not exceed ",
                        silero_vad::MAX_DURATION_MS,
                        " ms, got ",
                        value);
    }
}

}  // namespace ov::genai
