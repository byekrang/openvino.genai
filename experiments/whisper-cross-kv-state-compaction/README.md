# Whisper cross-KV VariableState batch compaction (Candidate #3)

## EXPERIMENTAL — NOT FOR MERGE

Standalone experiments (not GenAI production code) that evaluated an alternative
to the filler-row approach used by the native multi-batch Whisper decoder.
Preserved so the mechanism, performance behaviour, and the blocking OpenVINO
Runtime problem can be inspected.

Base: `feature/whisper-multi-batch-support` final PR commit.

## What "cross-KV state compaction" is

In native multi-batch Whisper, when one audio finishes mid-round its decoder row
cannot simply be dropped: self-attention KV is reordered by `beam_idx`, but
**cross-attention KV is row-bound and is not gathered by `beam_idx`**. The PR
keeps the finished row alive as a "filler" driven with `decoder_start_token_id`.

Candidate #3 instead **physically drops** the finished row:
- self-attention KV: existing `beam_idx=[0,2]` machinery (unchanged);
- cross-attention KV: compacted along the batch axis at the finish event via the
  OpenVINO `VariableState` API (`query_state` → `get_state` → copy-out → gather
  survivors → `set_state`). No graph change.

`[A,B,C] -> [A,C]` : cross-KV `set_state` to rows {0,2}; self-KV `beam_idx=[0,2]`.

## It is functionally correct

On the real whisper-tiny decoder, compacting only the cross-attention states and
running at the reduced width is **bit-identical** to an isolated run of the
surviving rows (max |logit diff| = 0.000), and *cleaner* than the filler path
(which has ~1e-4 batch-numeric drift). A negative control (keeping the wrong
survivor) diverges by ~1e1, confirming cross-KV is load-bearing.

## Where it helped / lost (CPU, integrated C++ measurement)

Net decoder-round wall time, filler vs compact (whisper-tiny / distil-small.en):

| batch (finish steps) | shrinks | winner | net |
|---|---|---|---|
| B2 imbalanced (78, 5) | 1 | **compact** | -50 ms (tiny) / -115 ms (distil) |
| B4 staggered (78,48,12,5) | 3 | **compact** | -57 ms / -223 ms |
| B4 short (6,11,12,12) | 2 | filler | +41 ms / +48 ms |
| B8 mixed (7 finishes) | 7 | filler | +28 ms |

- Wins on **length-imbalanced / few-shrink** batches with long lingering.
- Loses on **short** rounds and **many-shrink** batches (compaction cost accumulates).
- Break-even is in the **tens of decode steps** and does not systematically fall
  with model size. There is **no reliable runtime signal** for the surviving
  rows' remaining length, so it cannot be gated selectively.
- GPU: **unmeasured / unknown**.

## Why it is blocked

`VariableState::set_state()` that **shrinks the batch (outermost) dimension**
**heap-corrupts the OpenVINO CPU plugin** on the second round
(`corrupted size vs. prev_size` / invalid free inside
`libopenvino_intel_cpu_plugin.so`), and `reset_state()` + regrow fails with a
MatMul batch mismatch. A production Whisper pipeline runs many rounds, so this
blocks Candidate #3 regardless of its performance profile.

A minimal, GenAI-free reproducer is in `openvino-variable-state-repro/`
(reproduced on OpenVINO 2026.5.0-22942 and 2026.4.0). This is a Runtime problem
independent of this PR and is intended for the OpenVINO team.

## Files

- `integ.cpp` — integrated decode loop, filler vs compact, directly-measured round wall time (C++ VariableState surgery).
- `main.cpp` — single-round correctness (bit-identical) + shrink-cost microbench.
- `scale.cpp` — model-size scaling microbench (B=2/3/4).
- `e2e.py` — end-to-end workload harness (encoder + mel + greedy decode, both policies; dump/validate/bench).
- `scenario.py`, `run_e2e.sh`, `run_integ.sh` — isolated correctness + orchestration.
- `soak_probe.cpp`, `control.py`, `control_ds.py`, `caseB.py` — stability probes (batch-shrink corruption).
- `openvino-variable-state-repro/` — the standalone OpenVINO Runtime reproducer.
- `audio/` — small real-content length variants for the workload harness.

## How to run

Requires an OpenVINO 2026.5 build + a stateful Whisper decoder IR
(`optimum-cli export openvino --model openai/whisper-tiny <dir>`). The scripts
have machine-specific paths (model dir, `/tmp/cand3_audio`) — adjust them.

```
# C++ microbenchmarks / integrated A/B (link against libopenvino):
cmake -S . -B build -DOpenVINO_DIR=<ov>/cmake -GNinja && ninja -C build
./build/scale <decoder.xml> 1500 3 60           # shrink cost + per-step saving
./build/integ <decoder.xml> <enc_dir> "m1,m2" 1 # integrated round, filler vs compact

# Python workload harness:
python e2e.py <whisper_model_dir> validate       # transcripts match pipeline
python e2e.py <whisper_model_dir> bench          # per-batch A/B

# Stability reproducer:
cd openvino-variable-state-repro && OV=<ov> ./build.sh && ./repro <decoder.xml> shrink 3
```
