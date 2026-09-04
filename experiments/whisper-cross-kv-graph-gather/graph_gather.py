"""Candidate #4 prototype: insert Gather(beam_idx, axis=0) after each
cross-attention ReadValue in the Whisper decoder graph, so cross-KV is
reordered/compacted by beam_idx every step (like self-KV). Demonstrates:
  1) functional correctness (identity beam_idx -> logits identical to baseline);
  2) the per-step cost of the cross-KV Gather even when beam_idx is identity.

Cross-KV ReadValues are identified STRUCTURALLY: a state is self-attention iff
its Assign's data path concatenates the previous ReadValue (growing cache);
cross-attention states have no such self-concat.
"""
import sys, time, statistics as st, numpy as np
import openvino as ov
import openvino.opset13 as opset

MODEL = sys.argv[1]
E = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
core = ov.Core()

def classify_cross(model):
    """Return set of variable_ids that are cross-attention (no self-concat)."""
    def has_self_concat(assign_node, depth=8, seen=None):
        if seen is None: seen = set()
        if depth == 0: return False
        for i in range(assign_node.get_input_size()):
            src = assign_node.input_value(i).get_node()
            if src.get_type_name() == "Concat":
                return True
            if id(src) not in seen:
                seen.add(id(src))
                if has_self_concat(src, depth-1, seen):
                    return True
        return False
    cross = set()
    for s in model.get_sinks():
        vid = s.get_variable_id()
        if not has_self_concat(s):
            cross.add(vid)
    return cross

def build_gathered_model(path):
    model = core.read_model(path)
    cross = classify_cross(model)
    # locate beam_idx parameter
    beam = next(p for p in model.get_parameters() if p.get_friendly_name() == "beam_idx"
                or "beam_idx" in p.get_friendly_name())
    n_inserted = 0
    for op in model.get_ops():
        if op.get_type_name() != "ReadValue":
            continue
        if op.get_variable_id() not in cross:
            continue
        out = op.output(0)
        consumers = list(out.get_target_inputs())          # capture BEFORE inserting
        gather = opset.gather(out, beam.output(0),
                              opset.constant(0, ov.Type.i64))
        for inp in consumers:
            inp.replace_source_output(gather.output(0))
        n_inserted += 1
    model.validate_nodes_and_infer_types()
    return model, n_inserted, len(cross)

def prep(req, B, P, D):
    req.set_tensor("encoder_hidden_states", ov.Tensor((0.1*np.ones((B,E,D))).astype(np.float32)))
    req.set_tensor("input_ids", ov.Tensor(np.ones((B,P),dtype=np.int64)))
    req.set_tensor("beam_idx", ov.Tensor(np.arange(B,dtype=np.int32)))
    req.infer()

def per_step(cm, B, D, reps=60):
    r = cm.create_infer_request(); prep(r, B, 4, D)
    tok = np.full((B,1),100,dtype=np.int64); bi = np.arange(B,dtype=np.int32)  # IDENTITY beam_idx
    ts=[]
    for s in range(reps+5):
        r.set_tensor("encoder_hidden_states", ov.Tensor((0.1*np.ones((B,E,D))).astype(np.float32)))
        r.set_tensor("input_ids", ov.Tensor(tok)); r.set_tensor("beam_idx", ov.Tensor(bi))
        a=time.perf_counter(); r.infer(); b=time.perf_counter()
        if s>=5: ts.append((b-a)*1000)
    return st.median(ts), r

base = core.read_model(MODEL)
D = base.input("encoder_hidden_states").get_partial_shape()[2].get_length()
cm_base = core.compile_model(base, "CPU")
gm, n_ins, n_cross = build_gathered_model(MODEL)
cm_gath = core.compile_model(gm, "CPU")
print(f"model D={D} cross_states={n_cross} gathers_inserted={n_ins}")

# correctness: identity beam_idx -> logits must match baseline
def first_logits(cm, B):
    r=cm.create_infer_request(); prep(r,B,4,D)
    return r.get_tensor("logits").data.copy()
for B in (1,3):
    lb=first_logits(cm_base,B); lg=first_logits(cm_gath,B)
    print(f"  B={B} max|logits_gathered - baseline| = {np.max(np.abs(lb-lg)):.3e}")

print("\n== per-step decode cost (IDENTITY beam_idx) ==")
print(f"{'B':>3} {'baseline':>10} {'gathered':>10} {'overhead':>10}")
for B in (1,2,4,8):
    tb,_=per_step(cm_base,B,D); tg,_=per_step(cm_gath,B,D)
    print(f"{B:>3} {tb:>10.3f} {tg:>10.3f} {tg-tb:>+10.3f} ms/step  ({100*(tg-tb)/tb:+.1f}%)")
