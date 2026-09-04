// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// EXPERIMENTAL (not for merge) — see header.

#include "cross_kv_beam_gather.hpp"

#include <string>
#include <unordered_set>

#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/util/assign_base.hpp"
#include "openvino/op/util/read_value_base.hpp"

namespace ov::genai {
namespace {

// Iterative reverse-reachability: all ancestor nodes of `start` (inclusive).
std::unordered_set<const ov::Node*> collect_ancestors(const std::shared_ptr<ov::Node>& start) {
    std::unordered_set<const ov::Node*> seen;
    std::vector<std::shared_ptr<ov::Node>> stack{start};
    while (!stack.empty()) {
        auto n = stack.back(); stack.pop_back();
        for (size_t i = 0; i < n->get_input_size(); ++i) {
            auto src = n->input_value(i).get_node_shared_ptr();
            if (seen.insert(src.get()).second)
                stack.push_back(src);
        }
    }
    return seen;
}

bool set_contains_variable_readvalue(const std::unordered_set<const ov::Node*>& nodes,
                                     const std::string& variable_id) {
    for (const auto* n : nodes)
        if (auto rv = dynamic_cast<const ov::op::util::ReadValueBase*>(n))
            if (rv->get_variable_id() == variable_id)
                return true;
    return false;
}

// Self-attention iff a Concat lies on a path from the state's OWN ReadValue to
// its Assign (the growth recurrence past_kv = Concat(ReadValue(v), new_kv)):
// i.e. some Concat is an ancestor of the Assign AND the own ReadValue is an
// ancestor of that Concat. Cross-attention KV has no such growing Concat.
bool is_self_attention(const std::shared_ptr<ov::Node>& assign, const std::string& vid) {
    auto anc = collect_ancestors(assign);
    for (const auto* n : anc) {
        if (std::string(n->get_type_name()) != "Concat")
            continue;
        // BFS up from this Concat (need a shared_ptr; find it among the graph via inputs)
        auto concat = std::const_pointer_cast<ov::Node>(n->shared_from_this());
        if (set_contains_variable_readvalue(collect_ancestors(concat), vid))
            return true;
    }
    return false;
}

}  // namespace

void insert_cross_kv_beam_gather(const std::shared_ptr<ov::Model>& model) {
    // beam_idx must exist (stateful decoder contract).
    ov::Output<ov::Node> beam_idx;
    bool found_beam = false;
    for (const auto& p : model->get_parameters()) {
        if (p->get_friendly_name() == "beam_idx") { beam_idx = p->output(0); found_beam = true; break; }
    }
    OPENVINO_ASSERT(found_beam, "insert_cross_kv_beam_gather: no 'beam_idx' input; not a stateful decoder.");

    std::unordered_set<std::string> cross_ids, self_ids;
    for (const auto& sink : model->get_sinks()) {
        auto assign = std::dynamic_pointer_cast<ov::op::util::AssignBase>(sink);
        OPENVINO_ASSERT(assign, "insert_cross_kv_beam_gather: sink is not an Assign.");
        const std::string vid = assign->get_variable_id();
        if (is_self_attention(sink, vid))
            self_ids.insert(vid);
        else
            cross_ids.insert(vid);
    }

    OPENVINO_ASSERT(!cross_ids.empty(), "insert_cross_kv_beam_gather: found no cross-attention KV states.");
    // Whisper decoder topology: equal number of self- and cross-attention KV states.
    OPENVINO_ASSERT(self_ids.size() == cross_ids.size(),
                    "insert_cross_kv_beam_gather: unexpected topology (self=", self_ids.size(),
                    ", cross=", cross_ids.size(), "); refusing to transform.");

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    size_t inserted = 0;
    for (const auto& op : model->get_ops()) {
        auto rv = std::dynamic_pointer_cast<ov::op::util::ReadValueBase>(op);
        if (!rv || !cross_ids.count(rv->get_variable_id()))
            continue;
        auto out = op->output(0);
        OPENVINO_ASSERT(out.get_partial_shape().rank().is_static() &&
                            out.get_partial_shape().rank().get_length() == 4,
                        "insert_cross_kv_beam_gather: cross-KV state '", rv->get_variable_id(),
                        "' is not rank-4; refusing to transform.");
        auto consumers = out.get_target_inputs();  // capture before inserting Gather
        auto gather = std::make_shared<ov::op::v8::Gather>(out, beam_idx, axis);
        gather->set_friendly_name(rv->get_friendly_name() + "/cross_kv_beam_gather");
        for (auto& in : consumers)
            in.replace_source_output(gather->output(0));
        ++inserted;
    }
    OPENVINO_ASSERT(inserted == cross_ids.size(),
                    "insert_cross_kv_beam_gather: inserted ", inserted, " gathers for ",
                    cross_ids.size(), " cross states.");
    model->validate_nodes_and_infer_types();
}

}  // namespace ov::genai
