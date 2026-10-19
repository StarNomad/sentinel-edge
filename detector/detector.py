"""SENTINEL-EDGE detector: quantized PPE detection via ONNX Runtime.

Runs a YOLO-class single-stage detector exported to ONNX. On NPU hosts the
QNN execution provider is attempted first with CPU fallback; the provider
actually in use is recorded on the instance (never assumed). The Hailo-10H
INT8 path (Stage II) compiles the same ONNX graph with the Hailo Dataflow
Compiler.

Stage I model: keremberke/yolov8n-hard-hat-detection
  classes = ("Hardhat", "NO-Hardhat"), published mAP@0.5 = 0.836 (model card).
Hi-vis vest classes are a Stage II data round, not claimed here.
"""

import time

import numpy as np

try:
    import onnxruntime as ort
    _ORT = True
except ImportError:  # pragma: no cover
    _ORT = False


# Stage I: 2-class hard-hat model. Extended when vest data lands (Stage II).
CLASS_NAMES = ("Hardhat", "NO-Hardhat")
VIOLATION_CLASS = "NO-Hardhat"


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5):
    """Greedy NMS. boxes (n,4) xyxy, scores (n,) -> kept indices."""
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


class Detection:
    def __init__(self, xyxy, class_id, class_name, confidence):
        self.xyxy = xyxy              # (x1, y1, x2, y2) in pixels
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence


class Detector:
    def __init__(self, model_path: str, img_size: int = 640,
                 conf_threshold: float = 0.45, nms_threshold: float = 0.5,
                 providers=None, class_names: tuple = CLASS_NAMES):
        if not _ORT:
            raise RuntimeError("onnxruntime is not installed")
        self.class_names = tuple(class_names)
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.last_latency_ms = 0.0

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        wanted = providers or ["QNNExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(
                model_path, sess_options=opts, providers=wanted)
            self.provider_note = "requested: " + ",".join(wanted)
        except Exception as e:  # e.g. QNN init failure on this host
            self.session = ort.InferenceSession(
                model_path, sess_options=opts, providers=["CPUExecutionProvider"])
            self.provider_note = f"QNN unavailable ({type(e).__name__}); CPU fallback"
        self.active_providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR frame -> RGB, letterbox to img_size, normalize, NCHW."""
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = self.img_size / max(h, w)
        nh, nw = int(h * s), int(w * s)
        small = cv2.resize(rgb, (nw, nh))
        canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = small
        img = canvas.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))[None]

    def infer(self, frame: np.ndarray) -> list:
        blob = self.preprocess(frame)
        t0 = time.perf_counter()
        (out,) = self.session.run(None, {self.input_name: blob})
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return self._decode(out, frame.shape[1], frame.shape[0])

    def _decode(self, out: np.ndarray, w: int, h: int) -> list:
        """Decode YOLOv8-style (1, 4+nc, n) output with per-class NMS."""
        out = np.squeeze(out)
        if out.ndim != 2 or out.shape[0] < 5:
            return []
        s = self.img_size / max(h, w)
        boxes = out[:4].T / s          # (n,4) cx,cy,bw,bh in pixels
        scores = out[4:].T             # (n,nc)
        xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2,
                         boxes[:, 1] - boxes[:, 3] / 2,
                         boxes[:, 0] + boxes[:, 2] / 2,
                         boxes[:, 1] + boxes[:, 3] / 2], axis=1)
        dets = []
        for cid in range(min(scores.shape[1], len(self.class_names))):
            sc = scores[:, cid]
            mask = sc >= self.conf_threshold
            if not np.any(mask):
                continue
            for i in _nms(xyxy[mask], sc[mask], self.nms_threshold):
                j = int(np.where(mask)[0][i])
                x1, y1, x2, y2 = (float(v) for v in xyxy[j])
                dets.append(Detection((x1, y1, x2, y2), cid,
                                      self.class_names[cid], float(sc[j])))
        # Cross-class suppression: one head detected as both Hardhat and
        # NO-Hardhat must not become two tracks (phantom violation).
        # Keep the higher-confidence box (agnostic NMS).
        if len(dets) > 1:
            boxes = np.array([d.xyxy for d in dets])
            confs = np.array([d.confidence for d in dets])
            keep = _nms(boxes, confs, self.nms_threshold)
            dets = [dets[i] for i in keep]
        return dets
