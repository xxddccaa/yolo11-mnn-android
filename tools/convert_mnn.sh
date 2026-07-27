#!/usr/bin/env bash
# Convert exported YOLO11 ONNX models to MNN with 8-bit weight quantization.
#
# Weight-only INT8: halves model size, needs no calibration dataset, and keeps
# accuracy near-lossless (activations stay float). This is the robust mobile
# default vs full activation INT8 which needs a calibration set.
#
# MNN_SRC must point at an MNN source build that contains pymnn_build/MNNConvert
# and pymnn_build/tools/converter/libMNNConvertDeps.so. Set it in the env:
#
#   export MNN_SRC=/path/to/mnn/source-3.6.0
#   ./tools/convert_mnn.sh yolo11n.onnx yolo11s.onnx
#
set -euo pipefail

if [[ -z "${MNN_SRC:-}" ]]; then
  echo "ERROR: set MNN_SRC to your MNN source build dir (contains pymnn_build/)" >&2
  exit 1
fi

CONVERT="${MNN_SRC}/pymnn_build/MNNConvert"
export LD_LIBRARY_PATH="${MNN_SRC}/pymnn_build/tools/converter/:${MNN_SRC}/pymnn_build/:${LD_LIBRARY_PATH:-}"

if [[ ! -x "$CONVERT" ]]; then
  echo "ERROR: MNNConvert not found/executable at $CONVERT" >&2
  exit 1
fi

for onnx in "$@"; do
  mnn="${onnx%.onnx}.mnn"
  echo "[convert] $onnx -> $mnn (weightQuantBits=8)"
  "$CONVERT" \
    -f ONNX \
    --modelFile "$onnx" \
    --MNNModel "$mnn" \
    --weightQuantBits 8 \
    --weightQuantAsymmetric
  echo "[convert] done: $(ls -lh "$mnn" | awk '{print $5}')  $mnn"
done
