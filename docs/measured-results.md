# Measured results — Stage I

## Sustained end-to-end run (Dell 5441, 2026-10-12)

`pipeline/pipeline.py --bench-seconds 1800 --loop`, headless, QNN Execution
Provider active. Source: "Subway Builder video" (Wikimedia Commons,
CC BY 4.0, Colin Miller,
https://commons.wikimedia.org/wiki/File:Subway_Builder_video.webm) —
1527 frames @ 60fps, 1920x1162, ~25.5 s looped ~50x.
Timed: full iterations INCLUDING video capture/decode — capture +
preprocess + infer + track + events.

| Metric | Target | Measured | Status |
|---|---|---|---|
| Sustained FPS, end-to-end | >= 25 | 42.5 (76,564 frames / 1800.0 s) | PASS |
| Frame latency p95 (full loop) | <= 80 ms | 33.4 ms | PASS |

## Inference micro-benchmark (reference)

`demo/smoke.py`, 50 iterations, single 1280px image repeated:
74.1 FPS mean, p95 14.6 ms (QNN EP). CPU-only reference: 18.5 FPS.

## Cited, not measured

mAP (IoU 0.5) = 0.836 — keremberke/yolov8n-hard-hat-detection model card,
their val split, Ultralytics preprocessing. INT8 delta and power are
Stage II (UGen300 hardware).
