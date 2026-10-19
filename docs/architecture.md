# Architecture

```
 Camera (RTSP/USB, 720p25)
        |
        v
 +---------------------------+
 | UGen300 · Hailo-10H       |   Stage II target silicon
 | INT8 YOLO-class detector  |   (Stage I: Dell 5441, FP32 ONNX
 | hard-hat / no-hard-hat    |    via ONNX Runtime)
 +-------------+-------------+
               |
               v
 +---------------------------+
 | Host: tracker +           |   IoU tracker, missed-frame
 | PPE-state classifier      |   persistence -> per-track state:
 +-------------+-------------+   compliant / violation
               |
               v
 +---------------------------+
 | Event store (SQLite)      |   ts, zone, track_id, event,
 | sentinel.db               |   class, confidence, crop_path
 +-------------+-------------+   (batched commits)
               |
               v
 +---------------------------+
 | Shift analyst (on-host    |   reads EVENTS ONLY, never video
 | small LLM)                |   shift report + streak escalation
 +-------------+-------------+
               |
               v
     Dashboard + alerts  [PLANNED — Stage II]
```

Stage I classes: hard-hat, no-hard-hat (keremberke/yolov8n-hard-hat-detection,
published mAP@0.5 = 0.836, cited from the model card). Hi-vis vest classes are
a Stage II data round: same pipeline, extended class list.

## Data flow guarantees

1. Raw video never leaves the device. Only structured events leave it.
2. The analyst LLM's entire world is the event DB — it cannot see footage.
3. Every violation alert carries its evidence crop (on-device path).
4. The detector, tracker, store, and analyst communicate through frozen
   interfaces — each can be replaced without touching the others.

## Power story

- Accelerator: ~2.5 W typical (UGen300, Hailo-10H) — Stage II target.
- Stage I runs FP32 ONNX on the Dell 5441 (QNN EP attempted, CPU fallback);
  INT8 quantization lands with the Hailo Dataflow Compiler in Stage II.
- Whole node target: a fraction of a GPU appliance, deployable fanless.
