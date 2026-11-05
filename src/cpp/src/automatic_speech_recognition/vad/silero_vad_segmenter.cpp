// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/vad/silero_vad_segmenter.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "openvino/core/except.hpp"

namespace {

struct Segment {
    int64_t start = 0;
    int64_t end = 0;
};

int64_t samples_from_ms(size_t milliseconds) {
    return static_cast<int64_t>(ov::genai::silero_vad::SAMPLE_RATE / 1000) * static_cast<int64_t>(milliseconds);
}

int64_t floor_half(int64_t value) {
    return value >= 0 ? value / 2 : -((-value + 1) / 2);
}

}  // namespace

namespace ov::genai::silero_vad {

// Adapted from get_speech_timestamps_from_probs() in Silero VAD v6.2.3 (MIT License,
// Copyright (c) 2020-present Silero Team):
// https://github.com/snakers4/silero-vad/blob/5cd7945676eb32225748052e2e6a0580e4686a08/src/silero_vad/utils_vad.py#L421-L588
std::vector<SpeechSegment> get_speech_segments(const std::vector<float>& probabilities,
                                               size_t audio_length_samples,
                                               const SileroVADConfig& config) {
    config.validate();
    OPENVINO_ASSERT(audio_length_samples <= MAX_AUDIO_SAMPLES,
                    "Audio length ",
                    audio_length_samples,
                    " exceeds the supported maximum of ",
                    MAX_AUDIO_SAMPLES,
                    " samples");
    OPENVINO_ASSERT(probabilities.size() == (audio_length_samples + WINDOW_SAMPLES - 1) / WINDOW_SAMPLES,
                    "Expected one speech probability per ",
                    WINDOW_SAMPLES,
                    "-sample window of ",
                    audio_length_samples,
                    " samples, got ",
                    probabilities.size());

    const int64_t window = static_cast<int64_t>(WINDOW_SAMPLES);
    const int64_t audio_length = static_cast<int64_t>(audio_length_samples);
    const int64_t min_speech_samples = samples_from_ms(config.min_speech_duration_ms);
    const int64_t speech_pad_samples = samples_from_ms(config.speech_pad_ms);
    const double max_speech_samples = static_cast<double>(SAMPLE_RATE) * config.max_speech_duration_s -
                                      static_cast<double>(window) - 2.0 * static_cast<double>(speech_pad_samples);
    const int64_t min_silence_samples = samples_from_ms(config.min_silence_duration_ms);
    const int64_t min_silence_samples_at_max_speech = samples_from_ms(config.min_silence_at_max_speech_ms);
    const double threshold = config.threshold;
    const double neg_threshold = config.neg_threshold.value_or(std::max(threshold - 0.15, 0.01));

    bool triggered = false;
    std::vector<Segment> speeches;
    std::optional<Segment> current;
    int64_t temp_end = 0;
    int64_t prev_end = 0;
    int64_t next_start = 0;
    std::vector<std::pair<int64_t, int64_t>> possible_ends;

    const auto reset_split_state = [&]() {
        prev_end = 0;
        next_start = 0;
        temp_end = 0;
        possible_ends.clear();
    };

    for (size_t index = 0; index < probabilities.size(); ++index) {
        const double probability = probabilities[index];
        const int64_t cur_sample = window * static_cast<int64_t>(index);

        if (probability >= threshold && temp_end != 0) {
            const int64_t silence = cur_sample - temp_end;
            if (silence > min_silence_samples_at_max_speech) {
                possible_ends.emplace_back(temp_end, silence);
            }
            temp_end = 0;
            if (next_start < prev_end) {
                next_start = cur_sample;
            }
        }

        if (probability >= threshold && !triggered) {
            triggered = true;
            current = Segment{cur_sample, 0};
            continue;
        }

        if (triggered && static_cast<double>(cur_sample - current->start) > max_speech_samples) {
            if (config.use_max_poss_sil_at_max_speech && !possible_ends.empty()) {
                const auto longest = std::max_element(possible_ends.begin(),
                                                      possible_ends.end(),
                                                      [](const auto& lhs, const auto& rhs) {
                                                          return lhs.second < rhs.second;
                                                      });
                prev_end = longest->first;
                const int64_t duration = longest->second;
                current->end = prev_end;
                speeches.push_back(*current);
                current.reset();
                next_start = prev_end + duration;
                if (next_start < prev_end + cur_sample) {
                    current = Segment{next_start, 0};
                } else {
                    triggered = false;
                }
                reset_split_state();
            } else if (prev_end != 0) {
                current->end = prev_end;
                speeches.push_back(*current);
                current.reset();
                if (next_start < prev_end) {
                    triggered = false;
                } else {
                    current = Segment{next_start, 0};
                }
                reset_split_state();
            } else {
                current->end = cur_sample;
                speeches.push_back(*current);
                current.reset();
                reset_split_state();
                triggered = false;
                continue;
            }
        }

        if (probability < neg_threshold && triggered) {
            if (temp_end == 0) {
                temp_end = cur_sample;
            }
            const int64_t silence_now = cur_sample - temp_end;
            if (!config.use_max_poss_sil_at_max_speech && silence_now > min_silence_samples_at_max_speech) {
                prev_end = temp_end;
            }
            if (silence_now < min_silence_samples) {
                continue;
            }
            current->end = temp_end;
            if (current->end - current->start > min_speech_samples) {
                speeches.push_back(*current);
            }
            current.reset();
            reset_split_state();
            triggered = false;
            continue;
        }
    }

    if (current && audio_length - current->start > min_speech_samples) {
        current->end = audio_length;
        speeches.push_back(*current);
    }

    for (size_t index = 0; index < speeches.size(); ++index) {
        Segment& speech = speeches[index];
        if (index == 0) {
            speech.start = std::max<int64_t>(0, speech.start - speech_pad_samples);
        }
        if (index + 1 != speeches.size()) {
            Segment& next = speeches[index + 1];
            const int64_t silence = next.start - speech.end;
            if (silence < 2 * speech_pad_samples) {
                speech.end += floor_half(silence);
                next.start = std::max<int64_t>(0, next.start - floor_half(silence));
            } else {
                speech.end = std::min(audio_length, speech.end + speech_pad_samples);
                next.start = std::max<int64_t>(0, next.start - speech_pad_samples);
            }
        } else {
            speech.end = std::min(audio_length, speech.end + speech_pad_samples);
        }
    }

    std::vector<SpeechSegment> segments;
    segments.reserve(speeches.size());
    for (const Segment& speech : speeches) {
        segments.push_back({static_cast<size_t>(speech.start), static_cast<size_t>(speech.end)});
    }
    return segments;
}

}  // namespace ov::genai::silero_vad
