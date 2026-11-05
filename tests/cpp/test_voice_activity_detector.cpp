// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include "automatic_speech_recognition/vad/silero_vad_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/genai/automatic_speech_recognition/voice_activity_detector.hpp"
#include "openvino/runtime/properties.hpp"

using ov::genai::SileroVADConfig;
using ov::genai::SpeechSegment;
using ov::genai::VoiceActivityDetector;
using ov::genai::silero_vad::SileroVADModel;

namespace ov::genai {

inline void PrintTo(const SpeechSegment& segment, std::ostream* os) {
    *os << "[" << segment.start_sample << ", " << segment.end_sample << ")";
}

}  // namespace ov::genai

namespace {

using Segments = std::vector<SpeechSegment>;

// Fixtures are read from $TEST_MODELS_BASE_DIR/silero_vad_v6.2.3/: silero_vad_16k_sequence.onnx and
// silero_vad_openvino_16k.onnx from Silero VAD v6.2.3, and how_are_you_doing_today.wav from
// storage.openvinotoolkit.org. Tests skip when a fixture is unavailable unless GENAI_REQUIRE_SILERO_VAD=1, which
// CI sets so that model-backed coverage cannot be lost silently.
constexpr const char* kFixtureDirectory = "silero_vad_v6.2.3";
constexpr const char* kPerWindowModelFile = "silero_vad_openvino_16k.onnx";
constexpr const char* kSpeechFile = "how_are_you_doing_today.wav";
constexpr size_t kLeadingSilence = 20000;
constexpr size_t kGap = 16000;
constexpr size_t kRepeats = 8;

const Segments kMultiRegionDefault{{27168, 46560},
                                   {82464, 101856},
                                   {137760, 157152},
                                   {193568, 212960},
                                   {248864, 267744},
                                   {304160, 323552},
                                   {359456, 378848},
                                   {414752, 434144}};

uint32_t read_u32(const std::vector<char>& bytes, size_t offset) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) {
        value |= static_cast<uint32_t>(static_cast<unsigned char>(bytes.at(offset + i))) << (8 * i);
    }
    return value;
}

uint16_t read_u16(const std::vector<char>& bytes, size_t offset) {
    return static_cast<uint16_t>(static_cast<unsigned char>(bytes.at(offset)) |
                                 (static_cast<unsigned char>(bytes.at(offset + 1)) << 8));
}

std::vector<float> read_pcm16_mono_16k_wav(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    OPENVINO_ASSERT(bytes.size() >= 12 && std::memcmp(bytes.data(), "RIFF", 4) == 0 &&
                        std::memcmp(bytes.data() + 8, "WAVE", 4) == 0,
                    path,
                    " is not a RIFF/WAVE file");
    bool format_ok = false;
    for (size_t offset = 12; offset + 8 <= bytes.size();) {
        const std::string id(bytes.data() + offset, 4);
        const uint32_t size = read_u32(bytes, offset + 4);
        const size_t body = offset + 8;
        if (id == "fmt ") {
            format_ok = read_u16(bytes, body) == 1 && read_u16(bytes, body + 2) == 1 &&
                        read_u32(bytes, body + 4) == 16000 && read_u16(bytes, body + 14) == 16;
        } else if (id == "data") {
            OPENVINO_ASSERT(format_ok, path, " must be PCM16 mono 16 kHz");
            std::vector<float> audio(size / 2);
            for (size_t i = 0; i < audio.size(); ++i) {
                const auto sample = static_cast<int16_t>(read_u16(bytes, body + 2 * i));
                audio[i] = static_cast<float>(sample) / 32768.0f;
            }
            return audio;
        }
        offset = body + size + (size & 1);
    }
    OPENVINO_THROW(path, " has no data chunk");
}

std::vector<float> multi_region(const std::vector<float>& speech) {
    std::vector<float> audio(kLeadingSilence, 0.0f);
    for (size_t i = 0; i < kRepeats; ++i) {
        audio.insert(audio.end(), speech.begin(), speech.end());
        audio.insert(audio.end(), kGap, 0.0f);
    }
    return audio;
}

std::string construction_error(const std::filesystem::path& path, const ov::AnyMap& properties = {}) {
    try {
        VoiceActivityDetector detector(path, "CPU", properties);
    } catch (const ov::Exception& error) {
        return error.what();
    }
    return {};
}

class VoiceActivityDetectorModelTest : public testing::Test {
protected:
    void SetUp() override {
        const std::string missing = find_missing_fixture();
        if (missing.empty()) {
            return;
        }
        const char* required = std::getenv("GENAI_REQUIRE_SILERO_VAD");
        if (required != nullptr && std::string(required) == "1") {
            FAIL() << missing;
        }
        GTEST_SKIP() << missing;
    }

    std::string find_missing_fixture() {
        const char* base_dir = std::getenv("TEST_MODELS_BASE_DIR");
        if (base_dir == nullptr || *base_dir == '\0') {
            return "TEST_MODELS_BASE_DIR is not set";
        }
        m_model_dir = std::filesystem::path(base_dir) / kFixtureDirectory;
        for (const char* file : {ov::genai::silero_vad::SEQUENCE_MODEL_FILE, kPerWindowModelFile, kSpeechFile}) {
            if (!std::filesystem::is_regular_file(m_model_dir / file)) {
                return "Missing VAD test fixture " + (m_model_dir / file).string();
            }
        }
        return {};
    }

    std::vector<float> multi_region_speech() const {
        return multi_region(read_pcm16_mono_16k_wav(m_model_dir / kSpeechFile));
    }

    std::filesystem::path m_model_dir;
};

}  // namespace

TEST(VoiceActivityDetector, RejectsInvalidModelDirectory) {
    const auto root = std::filesystem::path(testing::TempDir());
    EXPECT_NE(construction_error(root / "silero_vad_missing_dir").find("must be a directory"), std::string::npos);
    const auto empty = root / "silero_vad_empty_dir";
    std::filesystem::create_directories(empty);
    EXPECT_NE(construction_error(empty).find(ov::genai::silero_vad::SEQUENCE_MODEL_FILE), std::string::npos);
}

TEST(VoiceActivityDetector, ValidatesConfigBeforeLoadingModel) {
    SileroVADConfig config;
    config.threshold = 2.0;
    const auto missing = std::filesystem::path(testing::TempDir()) / "silero_vad_missing_dir";
    EXPECT_NE(construction_error(missing, {ov::genai::silero_vad_config(config)}).find("threshold"),
              std::string::npos);
}

TEST_F(VoiceActivityDetectorModelTest, RejectsModelWithWrongIoContract) {
    const auto wrong = std::filesystem::path(testing::TempDir()) / "silero_vad_wrong_contract";
    std::filesystem::create_directories(wrong);
    std::filesystem::copy_file(m_model_dir / kPerWindowModelFile,
                               wrong / ov::genai::silero_vad::SEQUENCE_MODEL_FILE,
                               std::filesystem::copy_options::overwrite_existing);
    EXPECT_NE(construction_error(wrong).find("must have 3 inputs and 3 outputs"), std::string::npos);
}

TEST_F(VoiceActivityDetectorModelTest, EmptyAndShortInputsHaveNoSpeech) {
    VoiceActivityDetector detector(m_model_dir, "CPU");
    EXPECT_EQ(detector.detect({}), Segments{});
    for (const size_t length : {size_t{100}, size_t{511}, size_t{512}, size_t{513}}) {
        EXPECT_EQ(detector.detect(std::vector<float>(length, 0.0f)), Segments{}) << length;
    }
}

TEST_F(VoiceActivityDetectorModelTest, ModelReturnsOneProbabilityPerWindow) {
    SileroVADModel model(m_model_dir, "CPU", {});
    EXPECT_TRUE(model.get_speech_probabilities({}).empty());
    EXPECT_EQ(model.get_speech_probabilities(std::vector<float>(1, 0.0f)).size(), 1u);
    EXPECT_EQ(model.get_speech_probabilities(std::vector<float>(512, 0.0f)).size(), 1u);
    EXPECT_EQ(model.get_speech_probabilities(std::vector<float>(513, 0.0f)).size(), 2u);
    EXPECT_EQ(model.get_speech_probabilities(std::vector<float>(512 * 512, 0.0f)).size(), 512u);
    EXPECT_EQ(model.get_speech_probabilities(std::vector<float>(512 * 512 + 1, 0.0f)).size(), 513u);
}

TEST_F(VoiceActivityDetectorModelTest, RejectsNonFiniteAudio) {
    VoiceActivityDetector detector(m_model_dir, "CPU");
    for (const float value : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()}) {
        std::vector<float> audio(4096, 0.0f);
        audio[1000] = value;
        try {
            detector.detect(audio);
            ADD_FAILURE() << "expected an exception for sample " << value;
        } catch (const ov::Exception& error) {
            EXPECT_NE(std::string(error.what()).find("non-finite sample"), std::string::npos) << value;
            EXPECT_NE(std::string(error.what()).find("at index 1000"), std::string::npos) << value;
        }
    }
}

TEST_F(VoiceActivityDetectorModelTest, AcceptsCompileProperties) {
    VoiceActivityDetector detector(m_model_dir,
                                   "CPU",
                                   ov::genai::silero_vad_config(SileroVADConfig{}),
                                   ov::hint::inference_precision(ov::element::f32),
                                   ov::inference_num_threads(1));
    EXPECT_EQ(detector.detect(std::vector<float>(4096, 0.0f)), Segments{});
}

TEST_F(VoiceActivityDetectorModelTest, MultipleRegionsMatchUpstream) {
    VoiceActivityDetector detector(m_model_dir, "CPU");
    EXPECT_EQ(detector.detect(multi_region_speech()), kMultiRegionDefault);
}

TEST_F(VoiceActivityDetectorModelTest, ConfigurationMatchesUpstream) {
    SileroVADConfig config;
    config.max_speech_duration_s = 0.5;
    VoiceActivityDetector detector(m_model_dir, "CPU", ov::genai::silero_vad_config(config));
    EXPECT_EQ(detector.detect(multi_region_speech()),
              (Segments{{27168, 34560},   {34560, 41728},   {41728, 46560},   {82464, 89856},   {89856, 97024},
                        {97024, 101856},  {137760, 145152}, {145152, 152320}, {152320, 157152}, {193568, 200960},
                        {200960, 208128}, {208128, 212960}, {248864, 256256}, {256256, 263648}, {304160, 311552},
                        {311552, 318720}, {318720, 323552}, {359456, 366848}, {366848, 374016}, {374016, 378848},
                        {414752, 422144}, {422144, 429312}, {429312, 434144}}));
}

TEST_F(VoiceActivityDetectorModelTest, StateIsResetBetweenCalls) {
    const std::vector<float> audio = multi_region_speech();
    const std::vector<float> ends_in_speech(audio.begin(), audio.begin() + 80 * 512);

    SileroVADModel model(m_model_dir, "CPU", {});
    const std::vector<float> first = model.get_speech_probabilities(audio);
    model.get_speech_probabilities(ends_in_speech);
    EXPECT_EQ(model.get_speech_probabilities(audio), first);

    VoiceActivityDetector detector(m_model_dir, "CPU");
    EXPECT_EQ(detector.detect(audio), kMultiRegionDefault);
    detector.detect(ends_in_speech);
    EXPECT_EQ(detector.detect(audio), kMultiRegionDefault);
}
