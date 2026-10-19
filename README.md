# SENTINEL-EDGE

**Always-on hard-hat compliance vision for the 2.5-watt edge.**
ASUS UGen AI League Innovation Hackathon — Stage I entry
(Workplace AI · Battlefield Lightning · Global Online Track)

A quantized hard-hat detector that runs fully on-device.
Detection → tracking → PPE-state classification → event store → LLM shift analyst.
No cloud. No subscription. No footage ever leaves the site.

Stage I scope: hard-hat / no-hard-hat compliance. Hi-vis vest classes are a
Stage II data round and are not claimed here.

## Status

Stage I working code. The detector wrapper (ONNX Runtime, QNN attempted with
CPU fallback), tracker, event store, pipeline orchestration, and analyst are
real and run. Model: keremberke/yolov8n-hard-hat-detection
(classes Hardhat / NO-Hardhat; published mAP@0.5 = 0.836 on its val split —
their number, cited, not ours). The Hailo compile path and dashboard are
Stage II.

## Layout

```
detector/    ONNX Runtime detector wrapper (QNN EP attempted, CPU fallback;
             RGB preprocess, per-class + cross-class NMS)
pipeline/    Tracker (missed-frame persistence) + per-track PPE state +
             transition-debounced event logging with evidence crops +
             implemented streak-rule escalation
eventdb/     SQLite event store: run_id, timestamp, zone, track, event,
             class, confidence, crop path (batched commits)
analyst/     Rules-based shift report (trends vs previous, repeat tracks)
             + Ollama adapter; bring your own on-host LLM callable
docs/        Architecture diagram, measured results
demo/        smoke.py — on-device inference latency/FPS measurement
tests/       Logic tests: NMS, containment, tracker, debouncing, streak,
             metrics (run: python tests/test_sentinel.py)
```

## Quick start

From the repo root on the demo host:

```powershell
C:\tools\python310-x64\python.exe -m pip install -r requirements.txt

# live demo (webcam 0, or a video file)
$env:PYTHONPATH = (Get-Location)
C:\tools\python310-x64\python.exe -m pipeline.pipeline --model C:\tools\sentineledge\models\best.onnx --source 0

# sustained end-to-end benchmark (the deck's headline numbers)
C:\tools\python310-x64\python.exe -m pipeline.pipeline --model C:\tools\sentineledge\models\best.onnx --source C:\tools\sentineledge\models\bench.webm --bench-seconds 300 --loop
```

Events land in `sentinel.db` in the working directory; violation evidence
crops in `crops/`. The analyst reads that DB — never raw video.

## Reproducing the model

Weights are not in the repo (see `.gitignore`) — download and export:

```powershell
# 1. weights + sample image, from the model card
Invoke-WebRequest -Uri "https://huggingface.co/keremberke/yolov8n-hard-hat-detection/resolve/main/best.pt" -OutFile "best.pt"
Invoke-WebRequest -Uri "https://huggingface.co/keremberke/yolov8n-hard-hat-detection/resolve/main/thumbnail.jpg" -OutFile "test.jpg"

# 2. export to ONNX (ultralytics==8.4.175, opset 18, imgsz 640)
C:\tools\python310-x64\python.exe -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx', opset=18, imgsz=640)"
```

## Measurement methodology

Numbers in the proposal deck come from exactly three places:

- **Inference latency / FPS** — `demo/smoke.py`: full infer() path
  (preprocess + session.run + decode + NMS), timed over 50 iterations per
  image. The report states whether it was a single repeated image or a
  directory sweep.
- **Sustained end-to-end FPS / frame latency** — `pipeline/pipeline.py
  --bench-seconds 300 --loop`: headless full pipeline (capture + decode +
  preprocess + infer + track + events) on looped footage. The deck reports
  sustained FPS = frames/elapsed and the true full-loop p95.
- **Accuracy** — the model card's published mAP@0.5 = 0.836 (keremberke,
  yolov8n hard-hat val split), cited as theirs. Note the card's number was
  measured with Ultralytics' validation preprocessing (centered gray
  padding); our pipeline pads top-left with black — our own val-split run
  is Stage II work, pre-registered as a target.

Sustained-run footage: "Subway Builder video" (Wikimedia Commons,
CC BY 4.0, Colin Miller) —
https://commons.wikimedia.org/wiki/File:Subway_Builder_video.webm

Targets are pre-registered in `docs/measured-results.md`. Whatever the run
says — pass or miss — is what ships.

## Stage II plan (on UGen300 hardware)

1. HailoRT bring-up; compile the detector with the Hailo Dataflow Compiler.
2. INT8 calibration + accuracy recovery on a frozen val split.
3. Hi-vis vest data round (new classes, same pipeline).
4. Field soak on site-style footage; failure-analysis round.
5. Repo freeze with architecture, model info, and validation data.

## Model licensing

Stage I runs the off-the-shelf `keremberke/yolov8n-hard-hat-detection`
weights (Hugging Face), trained on the `keremberke/hard-hat-detection`
dataset. The model card states no license; the weights were trained with
Ultralytics YOLOv8 tooling (AGPL-3.0). Weight files are *not* redistributed
in this repo (see `.gitignore` — they download at build time), so the repo
itself contains no third-party model artifact. The code in this repo is the
entrant's own work, MIT licensed.

## AI-assistance disclosure

AI tools assisted drafting and code scaffolding for this entry. All
measurements, model runs, and the demo video are produced and verified by
the entrant.

## License

MIT — see LICENSE.
