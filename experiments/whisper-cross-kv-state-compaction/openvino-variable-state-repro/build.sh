#!/bin/bash
# Build the reproducer against an OpenVINO 2026.5 pip runtime.
# Usage: OV=/path/to/site-packages/openvino ./build.sh
set -e
: "${OV:?Set OV to the OpenVINO package directory (e.g. .../site-packages/openvino)}"
SO=$(ls "$OV"/libs/libopenvino.so.* | head -1)
g++ -std=c++17 -O2 repro.cpp -I"$OV/include" "$SO" -Wl,-rpath,"$OV/libs" -o repro
echo "built ./repro against $SO"
ldd ./repro | grep -q libopenvino_genai && echo "WARNING: genai linked" || echo "ok: no genai library linked"
