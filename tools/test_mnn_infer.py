#!/usr/bin/env python3
"""Self-test: verify the MNN weight-INT8 model matches the ONNX baseline.

Runs YOLO11 detection on a sample image through both onnxruntime (float ONNX,
the ground-truth baseline) and MNN (the quantized .mnn that ships in the APK),
using the identical letterbox / decode / NMS pipeline, then reports how well the
two detection sets agree (box IoU matching + score delta). This is the accuracy
gate before publishing.

Usage:
    python tools/test_mnn_infer.py \
        --onnx yolo11n.onnx --mnn yolo11n.mnn --img bus.jpg
"""
import argparse
import numpy as np
import cv2
import onnxruntime as ort
import MNN

CONF_THRES = 0.25
IOU_THRES = 0.45
INPUT = 640


def letterbox_square(img):
    """Pad to a square of side max(h,w) then resize to 640 (matches the app /
    wangzhaode/mnn-yolo decode). Returns the CHW float tensor and the scale."""
    ih, iw = img.shape[:2]
    length = max(ih, iw)
    scale = length / INPUT
    padded = np.zeros((length, length, 3), dtype=img.dtype)
    padded[:ih, :iw] = img
    resized = cv2.resize(padded, (INPUT, INPUT))
    blob = resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, /255
    blob = np.transpose(blob, (2, 0, 1))[None]  # HWC->CHW, add batch
    return np.ascontiguousarray(blob), scale


def decode(output, scale):
    """output: [84, 8400] -> list of (x0,y0,x1,y1,score,cls) in original image."""
    out = output.reshape(84, -1)
    boxes_xywh = out[:4]
    probs = out[4:]
    scores = probs.max(axis=0)
    classes = probs.argmax(axis=0)
    keep = scores > CONF_THRES
    cx, cy, w, h = boxes_xywh[:, keep]
    scores = scores[keep]
    classes = classes[keep]
    x0 = (cx - w / 2) * scale
    y0 = (cy - h / 2) * scale
    x1 = (cx + w / 2) * scale
    y1 = (cy + h / 2) * scale
    boxes = np.stack([x0, y0, x1, y1], axis=1)
    # class-agnostic NMS via cv2
    idxs = cv2.dnn.NMSBoxes(
        bboxes=[[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
        scores=scores.tolist(), score_threshold=CONF_THRES, nms_threshold=IOU_THRES,
    )
    idxs = np.array(idxs).flatten() if len(idxs) else np.array([], dtype=int)
    return [(*boxes[i], float(scores[i]), int(classes[i])) for i in idxs]


def run_onnx(path, blob):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: blob})[0][0]  # [84,8400]


def run_mnn(path, blob):
    interp = MNN.Interpreter(path)
    sess = interp.createSession({"backend": "CPU", "numThread": 4})
    inp = interp.getSessionInput(sess)
    interp.resizeTensor(inp, (1, 3, INPUT, INPUT))
    interp.resizeSession(sess)
    tmp = MNN.Tensor([1, 3, INPUT, INPUT], MNN.Halide_Type_Float,
                     blob, MNN.Tensor_DimensionType_Caffe)
    inp.copyFrom(tmp)
    interp.runSession(sess)
    out = interp.getSessionOutput(sess)
    host = MNN.Tensor(out.getShape(), MNN.Halide_Type_Float,
                      np.zeros(out.getShape(), dtype=np.float32),
                      MNN.Tensor_DimensionType_Caffe)
    out.copyToHostTensor(host)
    return np.array(host.getData(), dtype=np.float32).reshape(out.getShape())[0]


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-6)


def compare(det_onnx, det_mnn):
    matched, score_deltas = 0, []
    used = set()
    for da in det_onnx:
        best_j, best_iou = -1, 0.0
        for j, db in enumerate(det_mnn):
            if j in used or db[5] != da[5]:
                continue
            v = iou(da[:4], db[:4])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou > 0.5:
            matched += 1
            used.add(best_j)
            score_deltas.append(abs(da[4] - det_mnn[best_j][4]))
    return matched, score_deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--mnn", required=True)
    ap.add_argument("--img", required=True)
    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise SystemExit(f"cannot read image {args.img}")
    blob, scale = letterbox_square(img)

    det_onnx = decode(run_onnx(args.onnx, blob), scale)
    det_mnn = decode(run_mnn(args.mnn, blob), scale)

    print(f"ONNX detections: {len(det_onnx)}")
    print(f"MNN  detections: {len(det_mnn)}")
    matched, deltas = compare(det_onnx, det_mnn)
    total = max(len(det_onnx), 1)
    rate = matched / total
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    print(f"matched (IoU>0.5, same class): {matched}/{len(det_onnx)}  ({rate:.1%})")
    print(f"mean score delta on matches:   {mean_delta:.4f}")

    ok = rate >= 0.9 and mean_delta <= 0.05
    print("RESULT:", "PASS ✅" if ok else "FAIL ❌")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
