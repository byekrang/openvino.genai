# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import math
import os
import pathlib
import wave

import numpy as np
import openvino_genai as ov_genai
import pytest

pytestmark = pytest.mark.automatic_speech_recognition

SILERO_FIXTURE = "silero_vad_v6.2.3"
SPEECH_FILE = "how_are_you_doing_today.wav"

SILERO_MULTI_REGION = [(27168, 46560), (82464, 101856), (137760, 157152), (193568, 212960),
                       (248864, 267744), (304160, 323552), (359456, 378848), (414752, 434144)]


def fixture_path(*parts, required_by):
    base_dir = os.environ.get("TEST_MODELS_BASE_DIR")
    path = pathlib.Path(base_dir, *parts) if base_dir else None
    if path is None or not path.exists():
        message = f"Missing VAD test fixture {path}" if path else "TEST_MODELS_BASE_DIR is not set"
        if os.environ.get(required_by) == "1":
            pytest.fail(message)
        pytest.skip(message)
    return path


def multi_region_speech(required_by):
    with wave.open(str(fixture_path(SILERO_FIXTURE, SPEECH_FILE, required_by=required_by))) as file:
        speech = np.frombuffer(file.readframes(file.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    parts = [np.zeros(20000, dtype=np.float32)]
    for _ in range(8):
        parts += [speech, np.zeros(16000, dtype=np.float32)]
    return np.concatenate(parts)


def as_tuples(segments):
    return [(segment.start_sample, segment.end_sample) for segment in segments]


def test_configs_have_documented_defaults_and_validate():
    silero = ov_genai.SileroVADConfig()
    assert (silero.threshold, silero.neg_threshold, silero.speech_pad_ms) == (0.5, None, 30)
    assert math.isinf(silero.max_speech_duration_s)
    silero.threshold = 1.0
    with pytest.raises(RuntimeError, match="threshold"):
        silero.validate()


def test_speech_segment_is_a_read_only_value():
    segment = ov_genai.SpeechSegment(start_sample=160, end_sample=320)
    assert (segment.start_sample, segment.end_sample) == (160, 320)
    assert segment == ov_genai.SpeechSegment(160, 320)
    assert segment != ov_genai.SpeechSegment(160, 480)
    assert repr(segment) == "SpeechSegment(start_sample=160, end_sample=320)"
    with pytest.raises(AttributeError):
        segment.start_sample = 0
    with pytest.raises(TypeError):
        ov_genai.SpeechSegment()


def test_silero_detection_matches_upstream():
    model_dir = fixture_path(SILERO_FIXTURE, "silero_vad_16k_sequence.onnx", required_by="GENAI_REQUIRE_SILERO_VAD").parent
    detector = ov_genai.VoiceActivityDetector(model_dir, "CPU", silero_vad_config=ov_genai.SileroVADConfig())
    assert as_tuples(detector.detect(multi_region_speech("GENAI_REQUIRE_SILERO_VAD"))) == SILERO_MULTI_REGION

