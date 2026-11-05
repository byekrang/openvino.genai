// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::genai {

/**
 * @brief Speech region on the input audio timeline, as 16 kHz sample indices [start_sample, end_sample).
 */
struct SpeechSegment {
    size_t start_sample = 0;
    size_t end_sample = 0;

    friend bool operator==(const SpeechSegment& lhs, const SpeechSegment& rhs) {
        return lhs.start_sample == rhs.start_sample && lhs.end_sample == rhs.end_sample;
    }

    friend bool operator!=(const SpeechSegment& lhs, const SpeechSegment& rhs) {
        return !(lhs == rhs);
    }
};

/**
 * @brief Segmentation options for the Silero voice activity detector.
 *
 * Semantics follow upstream Silero VAD get_speech_timestamps() for 16 kHz audio and 512-sample windows.
 */
struct OPENVINO_GENAI_EXPORTS SileroVADConfig {
    /**
     * @brief Speech probability at or above which a window starts or continues speech.
     *
     * Must be in (0, 1). Default: 0.5.
     */
    double threshold = 0.5;

    /**
     * @brief Speech probability below which a window counts as silence.
     *
     * When unset, max(threshold - 0.15, 0.01) is used. When set, must be in (0, threshold].
     */
    std::optional<double> neg_threshold = std::nullopt;

    /**
     * @brief Minimum speech segment duration in milliseconds.
     *
     * Segments not longer than this are dropped. Default: 250.
     */
    size_t min_speech_duration_ms = 250;

    /**
     * @brief Minimum silence duration in milliseconds that ends a speech segment.
     *
     * Default: 100.
     */
    size_t min_silence_duration_ms = 100;

    /**
     * @brief Padding in milliseconds added to both sides of each speech segment.
     *
     * When neighbouring segments are closer than twice this value, the gap between them is split evenly.
     * Padding is clamped to the audio bounds. Default: 30.
     */
    size_t speech_pad_ms = 30;

    /**
     * @brief Maximum speech segment duration in seconds.
     *
     * Longer segments are split. Must be positive; infinity disables splitting. Default: infinity.
     */
    double max_speech_duration_s = std::numeric_limits<double>::infinity();

    /**
     * @brief Minimum silence duration in milliseconds considered as a split point for segments longer than
     * max_speech_duration_s.
     *
     * Default: 98.
     */
    size_t min_silence_at_max_speech_ms = 98;

    /**
     * @brief Whether segments longer than max_speech_duration_s are split at their longest silence.
     *
     * When false, they are split at the most recent silence. Default: true.
     */
    bool use_max_poss_sil_at_max_speech = true;

    /**
     * @brief Checks that all options are within their documented ranges.
     *
     * Millisecond durations must not exceed INT64_MAX / 64.
     * @throws ov::Exception if a value is out of range, NaN, or infinite where infinity is not allowed.
     */
    void validate() const;
};

/**
 * @brief Silero segmentation options for VoiceActivityDetector.
 *
 * Constructor-only property carrying a SileroVADConfig. Defaults are used when it is not set.
 */
static constexpr ov::Property<SileroVADConfig> silero_vad_config{"silero_vad_config"};

/**
 * @brief Standalone voice activity detector for mono 16 kHz audio.
 *
 * The model directory must contain silero_vad_16k_sequence.onnx from Silero VAD; versions 6.2.2 and 6.2.3
 * are validated. Recurrent state is reset on every detect() call.
 *
 * @note Not thread-safe. An instance may be reused for sequential detect() calls but must not be
 * called concurrently; use a separate instance per thread.
 */
class OPENVINO_GENAI_EXPORTS VoiceActivityDetector {
public:
    /**
     * @brief Constructs the voice activity detector from a model directory.
     * @param models_path Directory containing the detector model.
     * @param device Device to run inference on (e.g. "CPU").
     * @param properties ov::genai::silero_vad_config and OpenVINO compile properties.
     */
    VoiceActivityDetector(const std::filesystem::path& models_path,
                          const std::string& device,
                          const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    VoiceActivityDetector(const std::filesystem::path& models_path,
                          const std::string& device,
                          Properties&&... properties)
        : VoiceActivityDetector(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ~VoiceActivityDetector();

    /**
     * @brief Detects speech in mono 16 kHz float PCM.
     * @return Sorted, non-overlapping speech segments; empty when no speech is detected.
     * @throws ov::Exception if the audio contains NaN or infinite samples.
     */
    std::vector<SpeechSegment> detect(const std::vector<float>& audio);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ov::genai
