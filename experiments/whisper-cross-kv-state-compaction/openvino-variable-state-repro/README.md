# CPU plugin heap corruption when `VariableState::set_state()` shrinks batch dimension

## Environment
- OpenVINO **2026.5.0-22942-597262ef181** (pip `libopenvino.so.2650`), CPU plugin
- Linux x86-64, glibc malloc, g++ 13
- Also reproduced on OpenVINO **2026.4.0**
- Pure OpenVINO Runtime — **no `openvino_genai`, no tokenizer, no audio, no Python**

## Summary
A stateful encoder-decoder *decoder* model has dynamic-batch KV `VariableState`s
(`[?, heads, ?, head_size]`) and a `beam_idx` input. Self-attention KV is
reordered by `beam_idx` in-graph; cross-attention KV (sequence dim == encoder
length) is not, so to run at a smaller batch it is shrunk via
`VariableState::set_state()`.

Shrinking the **batch (outermost) dimension** of these states via `set_state`
and continuing inference fails in two ways:

- **Mode B (repeated fresh requests): heap corruption.** Round 1 succeeds;
  **round 2 aborts** with `corrupted size vs. prev_size while consolidating` /
  `free(): corrupted unsorted chunks`. Each round uses an independent
  `InferRequest`, so the corruption is via state shared through the compiled model.
- **Mode A (reset then regrow, one request): clean failure.** After
  `set_state` shrinks batch 4→3, `reset_state()` collapses states to the empty
  default `(0,H,0,64)`, but re-`prefill` at batch 4 fails:
  `MatMul ... Cant merge the first input dimension=4 with second input dimension=3`.

## Controls (only the batch-shrinking update fails)
| mode | operation | result |
|---|---|---|
| `none` | prefill B=4, infer B=4, no `set_state` | 3 rounds OK |
| `same` | prefill B=4, `set_state` **same** shape (B=4), infer B=4 | 3 rounds OK |
| `shrink` | prefill B=4, `set_state` **batch 4→3**, infer B=3 | **round 2 heap corruption** |

## Expected vs actual
- **Expected:** either the batch-shrinking `set_state` works consistently across
  independent requests / after `reset_state`, or it is rejected with a clean
  error. Same-shape `set_state` and no-`set_state` are stable.
- **Actual:** the batch-shrinking `set_state` corrupts the heap on the next
  request (Mode B) and cannot be regrown after `reset_state` (Mode A). Only the
  shape-changing (batch-shrinking) update fails.

## Model
Any stateful encoder-decoder decoder IR with dynamic-batch KV states and a
`beam_idx` input reproduces it. Example (used here):
```
optimum-cli export openvino --model openai/whisper-tiny whisper-tiny-ov
# -> whisper-tiny-ov/openvino_decoder_model.xml
```

## Build (OpenVINO 2026.5 runtime)
```
OV=/path/to/site-packages/openvino          # pip openvino 2026.5
g++ -std=c++17 -O2 repro.cpp \
    -I"$OV/include" "$OV/libs/libopenvino.so.2650" \
    -Wl,-rpath,"$OV/libs" -o repro
```
(Or `find_package(OpenVINO)` with `OpenVINO_DIR=$OV/cmake`.)

## Run (3 scenarios + controls)
```
DEC=whisper-tiny-ov/openvino_decoder_model.xml
./repro "$DEC" same 3         # control: same-shape set_state -> completes
./repro "$DEC" modeA          # shrink -> reset -> regrow -> MatMul 4 vs 3
./repro "$DEC" shrink 3       # repeated fresh-request shrink -> round 2 corrupts
```
For a native backtrace of the corruption:
```
gdb -q -batch -ex run -ex 'bt 20' --args ./repro "$DEC" shrink 3
```
(`MALLOC_CHECK_=3` yields the same abort message; it does not materially improve
the diagnostic. The gdb backtrace shows the invalid `free` and all frames inside
`libopenvino_intel_cpu_plugin.so`.)

## Captured results (2026.5.0-22942, CPU)
```
mode=same:  round 1 ok / round 2 ok / round 3 ok / COMPLETED
mode=none:  round 1 ok / round 2 ok / round 3 ok / COMPLETED

mode=modeA:
  [after prefill B=4]          cross[4,6,1500,64] x8  self[4,6,4,64] x8
  [after set_state cross 4->3] cross[3,6,1500,64] x8  self[4,6,4,64] x8
  [after infer B=3]            cross[3,6,1500,64] x8  self[3,6,5,64] x8
  [after reset_state]          self[0,6,0,64] x16
  regrow prefill B=4: FAILED ->
    [CPU] MatMul node 'MatMul_8924' ... Incompatible MatMul batch dimension.
    Cant merge the first input dimension=4 with second input dimension=3 at index=0

mode=shrink:
  round 1: ok
  round 2: corrupted size vs. prev_size while consolidating   (SIGABRT)
```
Native backtrace (separate run) shows the invalid `free` with every frame in
`libopenvino_intel_cpu_plugin.so`.

## Independence from GenAI
`repro.cpp` includes only `<openvino/openvino.hpp>`. `ldd ./repro` links only
`libopenvino.so.2650`, `libtbb.so.12`, and system libraries — **no
`libopenvino_genai`**. (Any "genai" substring in paths is only the venv
directory name, not a linked library.)

## Strongest justified conclusion
The failure is reproducible through the **public `ov::VariableState` API**
(`query_state` / `get_state` / `set_state` / `reset_state`) with pure OpenVINO
Runtime, and **localizes to the Intel CPU plugin** (`libopenvino_intel_cpu_plugin.so`).
Only the batch-dimension-shrinking `set_state` triggers it; same-shape
`set_state` and no-`set_state` are stable. The specific faulting source line is
not asserted here (the prebuilt plugin is stripped); a debug/ASan OpenVINO build
would pinpoint it.
