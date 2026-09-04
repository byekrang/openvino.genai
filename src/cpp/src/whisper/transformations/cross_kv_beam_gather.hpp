// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// EXPERIMENTAL (not for merge). Candidate #4: make Whisper cross-attention KV
// follow beam_idx like self-attention KV, by inserting
//   ReadValue(cross_KV) -> Gather(beam_idx, axis=0) -> existing consumers
// for every cross-attention state. A finished row can then be dropped by passing
// a compacting beam_idx (e.g. [0,2]) with no VariableState surgery.

#pragma once
#include <memory>
#include "openvino/core/model.hpp"

namespace ov::genai {
// Inserts the cross-KV beam_idx Gather. Asserts the expected Whisper stateful
// topology (equal self/cross KV states, a beam_idx input, rank-4 KV states).
void insert_cross_kv_beam_gather(const std::shared_ptr<ov::Model>& model);
}  // namespace ov::genai
