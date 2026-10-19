"""SENTINEL-EDGE pipeline: track, classify PPE state, log debounced events.

Per frame: detect -> associate detections to persistent tracks -> PPE state
per track -> log an event only on state transitions (not per frame).
Evidence crops are written for violations. DB commits are batched.

Stage I model classes: ("Hardhat", "NO-Hardhat"). A track's state comes
from its own detection class. The person-class path (5-class models,
Stage II) associates PPE boxes to person boxes by containment, not IoU:
a hard hat fully inside a person box has IoU ~0.08 but containment ~1.0.
"""

import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from detector.detector import Detector, CLASS_NAMES, VIOLATION_CLASS
from eventdb.eventdb import EventStore


def _box_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = _box_area(a) + _box_area(b) - inter
    return inter / ua if ua > 0 else 0.0


def containment(inner, outer) -> float:
    """Fraction of the inner box covered by the outer box."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = _box_area(inner)
    return inter / area if area > 0 else 0.0


class Tracker:
    """Greedy IoU tracker with missed-frame persistence. Tracks keep their
    ID for `max_missed` frames without a match before being dropped."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 10):
        self.tracks: dict[int, dict] = {}  # tid -> {"det": Detection, "missed": int}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def update(self, dets: list) -> dict[int, object]:
        for t in self.tracks.values():
            t["missed"] += 1
        assigned: dict[int, object] = {}
        used = set()
        for d in dets:
            best, best_iou = None, self.iou_threshold
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                v = iou(d.xyxy, t["det"].xyxy)
                if v > best_iou:
                    best, best_iou = tid, v
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {"det": d, "missed": 0}
            else:
                self.tracks[best] = {"det": d, "missed": 0}
            used.add(best)
            assigned[best] = d
        # age out stale tracks, but keep recently-missed ones for continuity
        for tid in [t for t, v in self.tracks.items()
                    if v["missed"] > self.max_missed]:
            del self.tracks[tid]
        return assigned


class Pipeline:
    def __init__(self, detector: Detector, store: EventStore,
                 zone: str = "default", crop_dir: str | Path = "crops",
                 transition_cooldown_s: float = 5.0, commit_every: int = 20):
        self.detector = detector
        self.store = store
        self.zone = zone
        self.crop_dir = Path(crop_dir)
        self.cooldown = transition_cooldown_s
        self.commit_every = commit_every
        # Person-class path only when the model actually has one (Stage II).
        self.person_class = ("person" if "person" in detector.class_names
                             else None)
        self.tracker = Tracker()
        self.last_state: dict[int, str] = {}
        self.last_event_t: dict[int, float] = {}
        self.violation_times: dict[int, deque] = {}
        self.run_id = uuid.uuid4().hex[:8]
        self._pending_commits = 0
        self.fps_ema = 0.0

    @staticmethod
    def _state_for(det) -> str:
        if det.class_name == VIOLATION_CLASS:
            return "violation"
        if det.class_name in CLASS_NAMES:
            return "compliant"
        return "uncertain"

    def _ppe_state_person_path(self, dets, track_boxes):
        """5-class path: PPE box contained (>50%) in a person box associates."""
        state, conf = {}, {}
        for tid, tb in track_boxes.items():
            tags = []
            for d in dets:
                if d.class_name == "person":
                    continue
                if containment(d.xyxy, tb) > 0.5:
                    tags.append(d)
            if any(d.class_name == VIOLATION_CLASS for d in tags):
                state[tid] = "violation"
            elif tags:
                state[tid] = "compliant"
            else:
                state[tid] = "uncertain"
            conf[tid] = max((d.confidence for d in tags), default=0.0)
        return state, conf

    def _maybe_log(self, tid: int, state: str, confidence: float,
                   frame, box) -> None:
        """Log only on state transitions (with cooldown against flapping)."""
        now = time.time()
        prev = self.last_state.get(tid)
        if state == prev:
            return
        if (prev is not None
                and now - self.last_event_t.get(tid, 0) < self.cooldown):
            return
        crop_path = None
        if state == "violation":
            self.crop_dir.mkdir(parents=True, exist_ok=True)
            x1, y1, x2, y2 = (int(v) for v in box)
            h, w = frame.shape[:2]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size:
                crop_path = str(self.crop_dir / f"{int(now)}_{tid}.jpg")
                cv2.imwrite(crop_path, crop)
        self.store.log(self.run_id, self.zone, tid, state, state,
                       confidence, crop_path)
        self.last_state[tid] = state
        self.last_event_t[tid] = now
        self._pending_commits += 1
        if state == "violation":
            self._check_streak(tid, now)
        if self._pending_commits >= self.commit_every:
            self.store.commit()
            self._pending_commits = 0

    def _check_streak(self, tid: int, now: float) -> None:
        """The streak rule, implemented: 3+ violation events for one track
        within 10 minutes escalates immediately."""
        dq = self.violation_times.setdefault(tid, deque())
        dq.append(now)
        while dq and now - dq[0] > 600:
            dq.popleft()
        if len(dq) >= 3:
            self.violation_times[tid] = deque()  # fire once per streak
            self.store.log(self.run_id, self.zone, tid, "escalation",
                           "streak-rule", 1.0, None)
            self._pending_commits += 1
            print(f"ALERT: streak rule — track {tid} in zone "
                  f"'{self.zone}': 3+ violations in 10 min")

    def run(self, source=0, show: bool = True,
            bench_seconds: float | None = None, loop: bool = False):
        """Run the live loop.

        bench_seconds: headless sustained measurement — no rendering, times
        every full iteration INCLUDING video capture/decode, loops the source
        if `loop`, and prints a BENCH summary at the end.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video source {source!r}")
        bench = bench_seconds is not None
        if bench:
            print(f"BENCH: {bench_seconds:g}s headless, loop={loop}, "
                  f"show=False. Timing full iterations incl. capture.")
        else:
            print("SENTINEL-EDGE live. Press Q to quit.")
        frame_times: list[float] = []
        t_start = time.perf_counter()
        try:
            while True:
                ft0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    if bench and loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                t0 = time.perf_counter()
                dets = self.detector.infer(frame)
                if self.person_class:
                    persons = [d for d in dets
                               if d.class_name == self.person_class]
                    tracks = self.tracker.update(persons)
                    boxes = {tid: d.xyxy for tid, d in tracks.items()}
                    states, confs = self._ppe_state_person_path(dets, boxes)
                else:
                    # 2-class path: the PPE detection IS the track.
                    tracks = self.tracker.update(dets)
                    states = {tid: self._state_for(d)
                              for tid, d in tracks.items()}
                    confs = {tid: d.confidence for tid, d in tracks.items()}
                    boxes = {tid: d.xyxy for tid, d in tracks.items()}
                for tid, st in states.items():
                    if st != "uncertain":
                        self._maybe_log(tid, st, confs[tid],
                                        frame, boxes[tid])
                dt = time.perf_counter() - t0
                self.fps_ema = (1 / dt if self.fps_ema == 0
                                else 0.9 * self.fps_ema + 0.1 / dt)
                frame_times.append((time.perf_counter() - ft0) * 1000.0)
                if bench and (time.perf_counter() - t_start) >= bench_seconds:
                    break
                if show and not bench:
                    for tid, tb in boxes.items():
                        x1, y1, x2, y2 = (int(v) for v in tb)
                        color = {"violation": (0, 0, 255),
                                 "compliant": (0, 200, 0),
                                 "uncertain": (0, 200, 200)}[states[tid]]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID{tid} {states[tid]}",
                                    (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(
                        frame,
                        f"{self.fps_ema:.1f} FPS  infer "
                        f"{self.detector.last_latency_ms:.1f} ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
                    cv2.imshow("sentinel-edge", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                        break
        finally:
            cap.release()
            if not bench:
                cv2.destroyAllWindows()
            self.store.commit()
            self.store.close()
        if bench and frame_times:
            frame_times.sort()
            n = len(frame_times)
            elapsed = time.perf_counter() - t_start
            mean = sum(frame_times) / n
            print(f"BENCH frames={n} elapsed_s={elapsed:.1f} "
                  f"sustained_fps={n / elapsed:.1f} "
                  f"mean_ms={mean:.1f} p95_ms={frame_times[int(n * 0.95)]:.1f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ONNX detector path")
    ap.add_argument("--source", default="0", help="webcam index or video file")
    ap.add_argument("--zone", default="default")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--bench-seconds", type=float, default=None,
                    help="headless sustained measurement for S seconds")
    ap.add_argument("--loop", action="store_true",
                    help="with --bench-seconds, loop video at EOF")
    args = ap.parse_args()
    try:
        src = int(args.source)
    except ValueError:
        src = args.source
    det = Detector(args.model)
    print("providers:", det.active_providers, "|", det.provider_note)
    store = EventStore()
    Pipeline(det, store, zone=args.zone).run(
        source=src, show=not args.no_show and args.bench_seconds is None,
        bench_seconds=args.bench_seconds, loop=args.loop)
