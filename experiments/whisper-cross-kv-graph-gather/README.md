# Whisper cross-KV graph Gather(beam_idx) (Candidate #4)

## EXPERIMENTAL — NOT FOR MERGE

A minimal graph transformation that makes Whisper **cross-attention KV follow
`beam_idx`** the same way self-attention KV already does, so a finished decoder
row can be dropped with `beam_idx` alone (no VariableState surgery).

Base: `feature/whisper-multi-batch-support` final PR commit.

## What it does

For every cross-attention KV `ReadValue`, insert:

```
ReadValue(cross_KV) -> Gather(indices=beam_idx, axis=0) -> existing consumers
```

Self-attention KV is already gathered by `beam_idx` in-graph; this makes
cross-attention KV behave identically. Dropping a row `[A,B,C] -> [A,C]` is then
just `beam_idx=[0,2]` with a width-2 inference.

Cross-attention states are identified structurally (iterative, no magic depth):
a state is *self*-attention iff a `Concat` lies on a path from the state's own
`ReadValue` to its `Assign` (the sequence-growth recurrence); otherwise it is
*cross*. The pass asserts the expected topology (a `beam_idx` input, equal
self/cross counts, rank-4 states) and fails loudly rather than silently
transforming unrelated states.

Production files:
- `src/cpp/src/whisper/transformations/cross_kv_beam_gather.{hpp,cpp}`
- `src/cpp/src/whisper/models/statefull_decoder.cpp` — 2-line experimental wiring
  (include + `insert_cross_kv_beam_gather(model);` after slice-before-matmul).

## It works

On whisper-tiny and distil-small.en:
- identity `beam_idx`: output **bit-identical** to baseline (0.000);
- drop-middle `[A,B,C] -> [A,C]` via `beam_idx=[0,2]`: **bit-identical** to an
  isolated `[A,C]` run (rowA 0.000, rowC 0.000);
- scalar beam permutation `beam_idx=[1,0]`: valid.

## Why it is rejected

The inserted `Gather` executes on **every decoder step, including the common
identity-`beam_idx` case** (normal transcription almost never reorders rows). It
copies the full cross-KV (~B×18 MB tiny / ~B×37 MB distil) every step, giving a
measured **~55–93% per-step CPU regression**:

| B | tiny base→xf | distil base→xf |
|--:|---|---|
| 1 | 3.9→6.6 ms (**+70%**) | 7.9→12.5 ms (**+59%**) |
| 2 | 4.2→8.2 ms (**+93%**) | 8.1→15.1 ms (**+86%**) |
| 4 | 5.1→8.6 ms (**+68%**) | 9.5→16.0 ms (**+69%**) |
| 8 | 6.6→10.9 ms (**+65%**) | 12.6→19.6 ms (**+55%**) |

Because the cost is unconditional and per-step, this is a net regression for
normal decoding; there is no conditional/identity fast path in this experiment
(adding one is out of scope). GPU: unmeasured.

## Files (this dir)

- `test_main.cpp` — applies the real C++ pass to a decoder IR and verifies
  identity / drop-middle / permutation correctness + per-step cost.
- `CMakeLists.txt` — builds the standalone verifier.
- `graph_gather.py`, `cand4_correct.py` — Python prototypes (transform +
  correctness + identity-Gather cost) that first demonstrated the behaviour.

## How to run the standalone verifier

```
cmake -S . -B build -DOpenVINO_DIR=<ov>/cmake -GNinja && ninja -C build
./build/cand4_test <decoder.xml> 1500
# or the Python prototype:
python graph_gather.py <decoder.xml> 1500
```
Requires an OpenVINO 2026.5 build and a stateful Whisper decoder IR
(`optimum-cli export openvino --model openai/whisper-tiny <dir>`).
