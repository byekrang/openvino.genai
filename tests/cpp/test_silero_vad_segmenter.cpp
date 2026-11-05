// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "automatic_speech_recognition/vad/silero_vad_segmenter.hpp"
#include "openvino/core/except.hpp"

using ov::genai::SileroVADConfig;
using ov::genai::SpeechSegment;
using ov::genai::silero_vad::get_speech_segments;
using ov::genai::silero_vad::MAX_AUDIO_SAMPLES;
using ov::genai::silero_vad::MAX_DURATION_MS;

namespace ov::genai {

inline void PrintTo(const SpeechSegment& segment, std::ostream* os) {
    *os << "[" << segment.start_sample << ", " << segment.end_sample << ")";
}

}  // namespace ov::genai

namespace {

constexpr size_t kWindow = ov::genai::silero_vad::WINDOW_SAMPLES;

std::vector<float> sequence(std::initializer_list<std::pair<float, size_t>> parts) {
    std::vector<float> probabilities;
    for (const auto& [value, count] : parts) {
        probabilities.insert(probabilities.end(), count, value);
    }
    return probabilities;
}

SileroVADConfig max_speech_config(double seconds, size_t min_silence_ms, bool longest_silence) {
    SileroVADConfig config;
    config.max_speech_duration_s = seconds;
    config.min_silence_duration_ms = min_silence_ms;
    config.use_max_poss_sil_at_max_speech = longest_silence;
    return config;
}

using Segments = std::vector<SpeechSegment>;

std::string exception_message(const std::vector<float>& probabilities, size_t audio_length) {
    try {
        get_speech_segments(probabilities, audio_length, SileroVADConfig{});
    } catch (const ov::Exception& error) {
        return error.what();
    }
    return {};
}

}  // namespace

TEST(SileroVADSegmenter, NoSpeechInputs) {
    EXPECT_EQ(get_speech_segments({}, 0, SileroVADConfig{}), Segments{});
    EXPECT_EQ(get_speech_segments(sequence({{0.0f, 10}}), 10 * kWindow, SileroVADConfig{}), Segments{});
}

TEST(SileroVADSegmenter, SingleBurstIsPaddedOnBothSides) {
    const auto probabilities = sequence({{0.0f, 5}, {0.9f, 20}, {0.0f, 15}});
    EXPECT_EQ(get_speech_segments(probabilities, 40 * kWindow, SileroVADConfig{}), (Segments{{2080, 13280}}));
}

TEST(SileroVADSegmenter, SpeechAtStartIsClampedAndTailEndsAtTrueLength) {
    const auto probabilities = sequence({{0.9f, 10}});
    EXPECT_EQ(get_speech_segments(probabilities, 10 * kWindow - 100, SileroVADConfig{}), (Segments{{0, 5020}}));
}

TEST(SileroVADSegmenter, SpeechMustBeLongerThanMinimum) {
    EXPECT_EQ(get_speech_segments(sequence({{0.0f, 3}, {0.9f, 7}, {0.0f, 10}}), 20 * kWindow, SileroVADConfig{}),
              Segments{});
    EXPECT_EQ(get_speech_segments(sequence({{0.0f, 3}, {0.9f, 8}, {0.0f, 10}}), 21 * kWindow, SileroVADConfig{}),
              (Segments{{1056, 6112}}));
    SileroVADConfig config;
    config.min_speech_duration_ms = 256;
    EXPECT_EQ(get_speech_segments(sequence({{0.0f, 3}, {0.9f, 8}, {0.0f, 10}}), 21 * kWindow, config), Segments{});
    EXPECT_EQ(get_speech_segments(sequence({{0.0f, 3}, {0.9f, 9}, {0.0f, 10}}), 22 * kWindow, config),
              (Segments{{1056, 6624}}));
    EXPECT_EQ(get_speech_segments(sequence({{0.0f, 10}, {0.9f, 8}}), 18 * kWindow, config), Segments{});
}

TEST(SileroVADSegmenter, ShortTailSpeechIsDropped) {
    const auto probabilities = sequence({{0.0f, 10}, {0.9f, 7}});
    EXPECT_EQ(get_speech_segments(probabilities, 17 * kWindow, SileroVADConfig{}), Segments{});
}

TEST(SileroVADSegmenter, ThresholdIsInclusive) {
    const auto probabilities = sequence({{0.0f, 2}, {0.5f, 12}, {0.0f, 10}});
    EXPECT_EQ(get_speech_segments(probabilities, 24 * kWindow, SileroVADConfig{}), (Segments{{544, 7648}}));
    const auto resumes_at_threshold = sequence({{0.9f, 10}, {0.1f, 3}, {0.5f, 1}, {0.1f, 2}, {0.9f, 10}, {0.0f, 10}});
    EXPECT_EQ(get_speech_segments(resumes_at_threshold, 36 * kWindow, SileroVADConfig{}), (Segments{{0, 13792}}));
}

TEST(SileroVADSegmenter, ProbabilitiesBetweenThresholdsKeepSpeech) {
    const auto probabilities = sequence({{0.9f, 10}, {0.4f, 10}, {0.2f, 10}});
    EXPECT_EQ(get_speech_segments(probabilities, 30 * kWindow, SileroVADConfig{}), (Segments{{0, 10720}}));
}

TEST(SileroVADSegmenter, ProbabilitiesBetweenThresholdsDoNotResetSilence) {
    const auto probabilities = sequence({{0.9f, 10}, {0.2f, 2}, {0.4f, 2}, {0.2f, 6}});
    EXPECT_EQ(get_speech_segments(probabilities, 20 * kWindow, SileroVADConfig{}), (Segments{{0, 5600}}));
}

TEST(SileroVADSegmenter, PauseEndingBeforeMinimumSilenceIsMerged) {
    const auto probabilities = sequence({{0.9f, 10}, {0.1f, 4}, {0.9f, 10}, {0.0f, 10}});
    EXPECT_EQ(get_speech_segments(probabilities, 34 * kWindow, SileroVADConfig{}), (Segments{{0, 12768}}));
}

TEST(SileroVADSegmenter, PauseReachingMinimumSilenceSplits) {
    const auto probabilities = sequence({{0.9f, 10}, {0.1f, 5}, {0.9f, 10}, {0.0f, 10}});
    EXPECT_EQ(get_speech_segments(probabilities, 35 * kWindow, SileroVADConfig{}),
              (Segments{{0, 5600}, {7200, 13280}}));
}

TEST(SileroVADSegmenter, PaddingIsSplitBetweenCloseSegments) {
    SileroVADConfig config;
    config.min_silence_duration_ms = 0;
    const auto probabilities = sequence({{0.9f, 10}, {0.1f, 1}, {0.9f, 10}, {0.0f, 5}});
    EXPECT_EQ(get_speech_segments(probabilities, 26 * kWindow, config), (Segments{{0, 5376}, {5376, 11232}}));
}

TEST(SileroVADSegmenter, NegativeThresholdControlsSilence) {
    const auto probabilities = sequence({{0.9f, 10}, {0.3f, 10}, {0.9f, 10}, {0.0f, 10}});
    EXPECT_EQ(get_speech_segments(probabilities, 40 * kWindow, SileroVADConfig{}),
              (Segments{{0, 5600}, {9760, 15840}}));
    SileroVADConfig config;
    config.neg_threshold = 0.25;
    EXPECT_EQ(get_speech_segments(probabilities, 40 * kWindow, config), (Segments{{0, 15840}}));
    const auto at_negative_threshold = sequence({{0.9f, 10}, {0.25f, 10}, {0.9f, 10}, {0.0f, 10}});
    EXPECT_EQ(get_speech_segments(at_negative_threshold, 40 * kWindow, config), (Segments{{0, 15840}}));
}

TEST(SileroVADSegmenter, MaxSpeechSplitsAtLongestSilence) {
    const auto probabilities = sequence({{0.9f, 12}, {0.1f, 6}, {0.9f, 6}, {0.1f, 4}, {0.9f, 12}, {0.0f, 20}});
    EXPECT_EQ(get_speech_segments(probabilities, 60 * kWindow, max_speech_config(1.0, 500, true)),
              (Segments{{0, 6624}, {8736, 24544}}));
    SileroVADConfig config = max_speech_config(1.0, 500, true);
    config.min_silence_at_max_speech_ms = 64;
    EXPECT_EQ(get_speech_segments(sequence({{0.9f, 12}, {0.1f, 2}, {0.9f, 20}, {0.0f, 10}}), 44 * kWindow, config),
              (Segments{{0, 15104}, {15104, 22528}}));
}

TEST(SileroVADSegmenter, MaxSpeechLegacySplitsAtLastSilence) {
    const auto probabilities = sequence({{0.9f, 12}, {0.1f, 6}, {0.9f, 6}, {0.1f, 4}, {0.9f, 12}, {0.0f, 20}});
    EXPECT_EQ(get_speech_segments(probabilities, 60 * kWindow, max_speech_config(1.0, 500, false)),
              (Segments{{0, 6624}, {8736, 20960}}));
    SileroVADConfig config = max_speech_config(1.0, 500, false);
    config.min_silence_at_max_speech_ms = 64;
    EXPECT_EQ(get_speech_segments(sequence({{0.9f, 12}, {0.1f, 3}, {0.9f, 20}, {0.0f, 10}}), 45 * kWindow, config),
              (Segments{{0, 15104}, {15104, 23040}}));
}

TEST(SileroVADSegmenter, MaxSpeechWithoutSilenceCutsAtCurrentWindow) {
    SileroVADConfig config;
    config.max_speech_duration_s = 1.0;
    const auto probabilities = sequence({{0.9f, 70}, {0.0f, 10}});
    EXPECT_EQ(get_speech_segments(probabilities, 80 * kWindow, config),
              (Segments{{0, 15104}, {15104, 30464}, {30464, 36320}}));
}

TEST(SileroVADSegmenter, RejectsProbabilityCountMismatch) {
    EXPECT_THROW(get_speech_segments(sequence({{0.9f, 3}}), 2 * kWindow, SileroVADConfig{}), ov::Exception);
    EXPECT_THROW(get_speech_segments(sequence({{0.9f, 2}}), 2 * kWindow + 1, SileroVADConfig{}), ov::Exception);
    EXPECT_NO_THROW(get_speech_segments(sequence({{0.9f, 3}}), 2 * kWindow + 1, SileroVADConfig{}));
}

TEST(SileroVADSegmenter, LargestMinimumSpeechDropsAllSpeech) {
    SileroVADConfig config;
    config.min_speech_duration_ms = MAX_DURATION_MS;
    const auto probabilities = sequence({{0.0f, 5}, {0.9f, 20}, {0.0f, 15}});
    EXPECT_EQ(get_speech_segments(probabilities, 40 * kWindow, config), Segments{});
}

TEST(SileroVADSegmenter, LargestMinimumSilenceKeepsSpeechToEnd) {
    SileroVADConfig config;
    config.min_silence_duration_ms = MAX_DURATION_MS;
    const auto probabilities = sequence({{0.0f, 5}, {0.9f, 20}, {0.0f, 15}});
    EXPECT_EQ(get_speech_segments(probabilities, 40 * kWindow, config), (Segments{{2080, 20480}}));
}

TEST(SileroVADSegmenter, LargestMaxSpeechSilenceFallsBackToHardCut) {
    SileroVADConfig config = max_speech_config(1.0, 500, true);
    config.min_silence_at_max_speech_ms = MAX_DURATION_MS;
    const auto probabilities = sequence({{0.9f, 12}, {0.1f, 6}, {0.9f, 6}, {0.1f, 4}, {0.9f, 12}, {0.0f, 20}});
    EXPECT_EQ(get_speech_segments(probabilities, 60 * kWindow, config), (Segments{{0, 15104}, {15104, 20960}}));
}

TEST(SileroVADSegmenter, LargestFiniteMaxSpeechDurationDoesNotSplit) {
    const auto probabilities = sequence({{0.9f, 70}, {0.0f, 10}});
    SileroVADConfig config;
    config.max_speech_duration_s = std::numeric_limits<double>::max();
    EXPECT_EQ(get_speech_segments(probabilities, 80 * kWindow, config), (Segments{{0, 36320}}));
    EXPECT_EQ(get_speech_segments(probabilities, 80 * kWindow, SileroVADConfig{}), (Segments{{0, 36320}}));
}

TEST(SileroVADSegmenter, AllDurationsAtLimit) {
    SileroVADConfig config;
    config.speech_pad_ms = MAX_DURATION_MS;
    config.min_silence_duration_ms = MAX_DURATION_MS;
    config.min_silence_at_max_speech_ms = MAX_DURATION_MS;
    config.max_speech_duration_s = std::numeric_limits<double>::max();
    const auto probabilities = sequence({{0.0f, 5}, {0.9f, 20}, {0.0f, 15}});
    EXPECT_EQ(get_speech_segments(probabilities, 40 * kWindow, config), (Segments{{0, 20480}}));
}

TEST(SileroVADSegmenter, RejectsAudioLongerThanSupported) {
    EXPECT_NE(exception_message({}, MAX_AUDIO_SAMPLES + 1).find("exceeds the supported maximum"), std::string::npos);
    EXPECT_NE(exception_message({}, MAX_AUDIO_SAMPLES).find("Expected one speech probability"), std::string::npos);
}

TEST(SileroVADConfigValidation, ThresholdMustBeFiniteInOpenUnitInterval) {
    EXPECT_NO_THROW(SileroVADConfig{}.validate());
    for (const double value : {0.0,
                               1.0,
                               -0.1,
                               1.5,
                               std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity()}) {
        SileroVADConfig config;
        config.threshold = value;
        EXPECT_THROW(config.validate(), ov::Exception) << value;
    }
}

TEST(SileroVADConfigValidation, NegativeThresholdMustBeFiniteAndNotAboveThreshold) {
    for (const double value : {0.0,
                               -0.1,
                               0.51,
                               std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity()}) {
        SileroVADConfig config;
        config.neg_threshold = value;
        EXPECT_THROW(config.validate(), ov::Exception) << value;
    }
    SileroVADConfig config;
    config.neg_threshold = config.threshold;
    EXPECT_NO_THROW(config.validate());
}

TEST(SileroVADConfigValidation, MaxSpeechDurationMustBePositiveOrInfinity) {
    for (const double value :
         {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), -std::numeric_limits<double>::infinity()}) {
        SileroVADConfig config;
        config.max_speech_duration_s = value;
        EXPECT_THROW(config.validate(), ov::Exception) << value;
    }
    for (const double value :
         {0.5, 30.0, std::numeric_limits<double>::max(), std::numeric_limits<double>::infinity()}) {
        SileroVADConfig config;
        config.max_speech_duration_s = value;
        EXPECT_NO_THROW(config.validate()) << value;
    }
}

TEST(SileroVADConfigValidation, DurationsAreBoundedToAvoidSampleOverflow) {
    EXPECT_EQ(MAX_DURATION_MS, static_cast<size_t>(144115188075855871ULL));
    for (size_t SileroVADConfig::*field : {&SileroVADConfig::min_speech_duration_ms,
                                           &SileroVADConfig::min_silence_duration_ms,
                                           &SileroVADConfig::speech_pad_ms,
                                           &SileroVADConfig::min_silence_at_max_speech_ms}) {
        SileroVADConfig config;
        config.*field = MAX_DURATION_MS;
        EXPECT_NO_THROW(config.validate());
        config.*field = MAX_DURATION_MS + 1;
        EXPECT_THROW(config.validate(), ov::Exception);
        config.*field = std::numeric_limits<size_t>::max();
        EXPECT_THROW(config.validate(), ov::Exception);
    }
}
