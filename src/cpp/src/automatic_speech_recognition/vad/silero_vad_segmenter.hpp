// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "openvino/genai/automatic_speech_recognition/voice_activity_detector.hpp"

namespace ov::genai::silero_vad {

constexpr size_t SAMPLE_RATE = 16000;
constexpr size_t WINDOW_SAMPLES = 512;

// Bounds keep doubled padding and position-plus-padding sums within int64 sample arithmetic.
constexpr size_t MAX_DURATION_MS = static_cast<size_t>(std::numeric_limits<int64_t>::max() / 64);
constexpr size_t MAX_AUDIO_SAMPLES = static_cast<size_t>(std::numeric_limits<int64_t>::max() / 4);

/**
 * @brief Port of upstream Silero VAD get_speech_timestamps_from_probs() for 16 kHz audio.
 *
 * Expects one probability per 512-sample window of audio_length_samples, the last window possibly partial.
 */
std::vector<SpeechSegment> get_speech_segments(const std::vector<float>& probabilities,
                                               size_t audio_length_samples,
                                               const SileroVADConfig& config);

}  // namespace ov::genai::silero_vad
