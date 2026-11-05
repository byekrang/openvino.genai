// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/vad/silero_vad_model.hpp"

#include <algorithm>
#include <array>
#include <memory>

#include "automatic_speech_recognition/vad/silero_vad_segmenter.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils.hpp"

namespace {

using ov::genai::silero_vad::CONTEXT_SAMPLES;
using ov::genai::silero_vad::STATE_SIZE;
using ov::genai::silero_vad::WINDOW_SAMPLES;

constexpr size_t FRAME_INPUT_SAMPLES = CONTEXT_SAMPLES + WINDOW_SAMPLES;
const ov::Shape STATE_SHAPE{1, 1, STATE_SIZE};

std::string describe_port(const ov::Output<ov::Node>& port) {
    return port.get_any_name() + " " + port.get_element_type().get_type_name() + port.get_partial_shape().to_string();
}

const ov::Output<ov::Node>& find_port(const std::vector<ov::Output<ov::Node>>& ports,
                                      const std::string& name,
                                      const std::filesystem::path& model_path) {
    const auto it = std::find_if(ports.begin(), ports.end(), [&name](const ov::Output<ov::Node>& port) {
        return port.get_names().count(name) > 0;
    });
    OPENVINO_ASSERT(it != ports.end(), "Silero VAD model ", model_path, " has no '", name, "' port");
    return *it;
}

void expect_port(const std::vector<ov::Output<ov::Node>>& ports,
                 const std::string& name,
                 const ov::PartialShape& shape,
                 const std::filesystem::path& model_path) {
    const ov::Output<ov::Node>& port = find_port(ports, name, model_path);
    OPENVINO_ASSERT(port.get_element_type() == ov::element::f32 && port.get_partial_shape().compatible(shape) &&
                        port.get_partial_shape().rank() == shape.rank(),
                    "Silero VAD model ",
                    model_path,
                    " port ",
                    describe_port(port),
                    " does not match the expected f32",
                    shape.to_string());
}

void validate_io_contract(const std::shared_ptr<ov::Model>& model, const std::filesystem::path& model_path) {
    OPENVINO_ASSERT(model->inputs().size() == 3 && model->outputs().size() == 3,
                    "Silero VAD model ",
                    model_path,
                    " must have 3 inputs and 3 outputs, got ",
                    model->inputs().size(),
                    " and ",
                    model->outputs().size());
    const ov::PartialShape state{1, 1, static_cast<int64_t>(STATE_SIZE)};
    const ov::PartialShape frames{ov::Dimension::dynamic(), static_cast<int64_t>(FRAME_INPUT_SAMPLES)};
    expect_port(model->inputs(), "input", frames, model_path);
    OPENVINO_ASSERT(find_port(model->inputs(), "input", model_path).get_partial_shape()[0].is_dynamic(),
                    "Silero VAD model ",
                    model_path,
                    " must have a dynamic sequence dimension on 'input'");
    expect_port(model->inputs(), "h", state, model_path);
    expect_port(model->inputs(), "c", state, model_path);
    expect_port(model->outputs(), "speech_probs", {ov::Dimension::dynamic()}, model_path);
    expect_port(model->outputs(), "hn", state, model_path);
    expect_port(model->outputs(), "cn", state, model_path);
}

}  // namespace

namespace ov::genai::silero_vad {

SileroVADModel::SileroVADModel(const std::filesystem::path& models_path,
                               const std::string& device,
                               const ov::AnyMap& properties) {
    OPENVINO_ASSERT(std::filesystem::is_directory(models_path),
                    "Silero VAD models_path must be a directory, got ",
                    models_path);
    const std::filesystem::path model_path = models_path / SEQUENCE_MODEL_FILE;
    OPENVINO_ASSERT(std::filesystem::is_regular_file(model_path),
                    "Silero VAD model file ",
                    SEQUENCE_MODEL_FILE,
                    " (validated with silero-vad 6.2.2 and 6.2.3) not found in ",
                    models_path);

    ov::Core& core = utils::singleton_core();
    const std::shared_ptr<ov::Model> model = core.read_model(model_path);
    validate_io_contract(model, model_path);

    ov::AnyMap compile_properties = properties;
    compile_properties.emplace(ov::hint::inference_precision.name(), ov::element::f32);
    ov::CompiledModel compiled = core.compile_model(model, device, compile_properties);
    utils::print_compiled_model_properties(compiled, "Silero VAD model");
    m_request = compiled.create_infer_request();
}

std::vector<float> SileroVADModel::get_speech_probabilities(const std::vector<float>& audio) {
    const size_t frame_count = (audio.size() + WINDOW_SAMPLES - 1) / WINDOW_SAMPLES;
    std::vector<float> probabilities;
    probabilities.reserve(frame_count);

    ov::Tensor hidden(ov::element::f32, STATE_SHAPE);
    ov::Tensor cell(ov::element::f32, STATE_SHAPE);
    std::fill_n(hidden.data<float>(), hidden.get_size(), 0.0f);
    std::fill_n(cell.data<float>(), cell.get_size(), 0.0f);
    std::array<float, CONTEXT_SAMPLES> context{};

    for (size_t first_frame = 0; first_frame < frame_count; first_frame += MAX_BLOCK_FRAMES) {
        const size_t block_frames = std::min(MAX_BLOCK_FRAMES, frame_count - first_frame);
        ov::Tensor input(ov::element::f32, {block_frames, FRAME_INPUT_SAMPLES});
        float* rows = input.data<float>();
        for (size_t frame = 0; frame < block_frames; ++frame) {
            float* row = rows + frame * FRAME_INPUT_SAMPLES;
            std::copy(context.begin(), context.end(), row);
            float* window = row + CONTEXT_SAMPLES;
            const size_t begin = (first_frame + frame) * WINDOW_SAMPLES;
            const size_t available = std::min(WINDOW_SAMPLES, audio.size() - begin);
            std::copy_n(audio.data() + begin, available, window);
            std::fill(window + available, window + WINDOW_SAMPLES, 0.0f);
            std::copy(window + WINDOW_SAMPLES - CONTEXT_SAMPLES, window + WINDOW_SAMPLES, context.begin());
        }

        m_request.set_tensor("input", input);
        m_request.set_tensor("h", hidden);
        m_request.set_tensor("c", cell);
        m_request.infer();

        const ov::Tensor speech_probs = m_request.get_tensor("speech_probs");
        OPENVINO_ASSERT(speech_probs.get_size() == block_frames,
                        "Silero VAD model returned ",
                        speech_probs.get_size(),
                        " probabilities for ",
                        block_frames,
                        " frames");
        const float* values = speech_probs.data<const float>();
        probabilities.insert(probabilities.end(), values, values + block_frames);

        ov::Tensor next_hidden(ov::element::f32, STATE_SHAPE);
        ov::Tensor next_cell(ov::element::f32, STATE_SHAPE);
        m_request.get_tensor("hn").copy_to(next_hidden);
        m_request.get_tensor("cn").copy_to(next_cell);
        hidden = next_hidden;
        cell = next_cell;
    }
    return probabilities;
}

}  // namespace ov::genai::silero_vad
