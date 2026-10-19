"""SENTINEL-EDGE logic tests. Run: python -m pytest tests/ -q
(or python tests/test_sentinel.py). No model, no camera needed."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detector.detector import _nms, Detector
from pipeline.pipeline import Pipeline, Tracker, containment, iou
from eventdb.eventdb import EventStore
from analyst.analyst import render_shift_report


class Det:
    def __init__(self, xyxy, class_id, class_name, confidence):
        self.xyxy = tuple(xyxy)
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence


class FakeDetector:
    def __init__(self, class_names=("Hardhat", "NO-Hardhat")):
        self.class_names = class_names
        self.last_latency_ms = 1.0
        self._dets = []

    def infer(self, frame):
        return list(self._dets)


def test_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
                     dtype=float)
    assert _nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]


def test_cross_class_nms_kills_phantom_violation():
    # one head, detected as both classes -> must decode to ONE detection
    det = Detector.__new__(Detector)
    det.img_size, det.conf_threshold, det.nms_threshold = 640, 0.45, 0.5
    det.class_names = ("Hardhat", "NO-Hardhat")
    n = 8400
    out = np.zeros((1, 6, n), dtype=np.float32)
    out[0, 0, 0], out[0, 1, 0], out[0, 2, 0], out[0, 3, 0] = 320, 320, 40, 40
    out[0, 4, 0] = 0.60   # Hardhat
    out[0, 5, 0] = 0.50   # NO-Hardhat, same box
    dets = det._decode(out, 640, 640)
    assert len(dets) == 1, f"phantom duplicate: {len(dets)}"
    assert dets[0].class_name == "Hardhat"  # higher confidence wins


def test_containment_beats_iou_for_hats():
    person = (100, 100, 200, 300)   # tall box
    hat = (140, 100, 170, 130)      # fully inside, top
    assert iou(hat, person) < 0.1, "sanity: IoU really is tiny here"
    assert containment(hat, person) > 0.9


def test_tracker_persists_through_missed_frames():
    t = Tracker(iou_threshold=0.3, max_missed=10)
    d = Det((0, 0, 10, 10), 0, "Hardhat", 0.9)
    a1 = t.update([d])
    for _ in range(5):  # missed frames: no detections
        t.update([])
    a2 = t.update([Det((1, 1, 11, 11), 0, "Hardhat", 0.9)])
    assert set(a1) == set(a2), "track ID must survive missed frames"


def test_tracker_drops_after_max_missed():
    t = Tracker(iou_threshold=0.3, max_missed=3)
    a1 = t.update([Det((0, 0, 10, 10), 0, "Hardhat", 0.9)])
    for _ in range(4):
        t.update([])
    a2 = t.update([Det((0, 0, 10, 10), 0, "Hardhat", 0.9)])
    assert set(a1) != set(a2), "stale track must be dropped"


def test_events_debounced_not_per_frame():
    fd = FakeDetector()
    fd._dets = [Det((0, 0, 10, 10), 1, "NO-Hardhat", 0.9)]
    store = EventStore(":memory:")
    p = Pipeline(fd, store, zone="z", crop_dir="/tmp/se-crops",
                 transition_cooldown_s=3600)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(1000):  # 1000 identical violation frames
        dets = fd.infer(frame)
        tracks = p.tracker.update(dets)
        for tid, d in tracks.items():
            p._maybe_log(tid, "violation", d.confidence, frame, d.xyxy)
    n = store.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert n == 1, f"expected 1 debounced event, got {n}"
    store.close()


def test_streak_rule_fires():
    fd = FakeDetector()
    store = EventStore(":memory:")
    p = Pipeline(fd, store, zone="z", crop_dir="/tmp/se-crops")
    now = time.time()
    p._check_streak(7, now)
    p._check_streak(7, now + 60)
    p._check_streak(7, now + 120)
    n = store.db.execute(
        "SELECT COUNT(*) FROM events WHERE event='escalation'").fetchone()[0]
    assert n == 1, "3 violations in 10 min must escalate"
    store.close()


def test_violations_per_hour_uses_observed_span():
    store = EventStore(":memory:")
    for _ in range(3):
        store.log("run1", "z", 1, "violation", "NO-Hardhat", 0.9)
    store.commit()
    s = store.shift_summary(hours=8.0)
    # 3 violations observed within seconds, not spread over 8 hours
    assert s["zones"][0]["violations_per_hour"] > 100, s["zones"][0]
    assert s["observed_hours"] < 0.01
    store.close()


def test_rules_report_trends_vs_previous():
    s = {"observed_hours": 0.1, "zones": [
        {"zone": "z", "events": 4, "violations": 3, "compliant": 1,
         "escalations": 0, "transition_compliance_rate": 0.25,
         "violations_per_hour": 30.0}], "repeat_tracks": []}
    prev = {"zones": [
        {"zone": "z", "events": 4, "violations": 1, "compliant": 3,
         "escalations": 0, "transition_compliance_rate": 0.75,
         "violations_per_hour": 10.0}]}
    rep = render_shift_report(s, previous=prev)
    assert "vs last shift" in rep, rep


def test_person_path_states():
    fd = FakeDetector(class_names=("person", "Hardhat", "NO-Hardhat"))
    store = EventStore(":memory:")
    p = Pipeline(fd, store, zone="z", crop_dir="/tmp/se-crops")
    assert p.person_class == "person"
    person = (100, 100, 200, 300)
    hat = (140, 100, 170, 130)
    dets = [Det(person, 0, "person", 0.95), Det(hat, 1, "Hardhat", 0.8)]
    states, confs = p._ppe_state_person_path(
        dets, {1: person})
    assert states[1] == "compliant", states
    assert confs[1] == 0.8
    store.close()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
