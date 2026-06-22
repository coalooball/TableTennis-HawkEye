from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan2, degrees
from typing import Iterable

import numpy as np


COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_ELBOW = 7
COCO_RIGHT_ELBOW = 8
COCO_LEFT_WRIST = 9
COCO_RIGHT_WRIST = 10
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12


@dataclass
class ActionResult:
    label: str
    confidence: float
    reason: str


class SimpleActionRecognizer:
    """Rule-based action recognizer from YOLO Pose keypoints.

    This is intentionally small and deterministic. It classifies broad movement
    states from COCO 17-keypoint skeletons; domain-specific actions should be
    trained later with a temporal model.
    """

    def __init__(self, history_size: int = 8, min_keypoint_conf: float = 0.35):
        self.history: deque[np.ndarray] = deque(maxlen=history_size)
        self.min_keypoint_conf = min_keypoint_conf

    def update(self, keypoints: np.ndarray | None) -> ActionResult:
        if keypoints is None or keypoints.shape[0] < 17:
            return ActionResult("no_person", 0.0, "no valid person skeleton")

        self.history.append(keypoints)
        visible = self._visible_points(keypoints)
        if visible < 6:
            return ActionResult("uncertain", 0.2, "too few reliable keypoints")

        if self._both_wrists_above_shoulders(keypoints):
            return ActionResult("hands_up", 0.86, "both wrists are above shoulders")

        if self._single_arm_raised(keypoints):
            return ActionResult("one_hand_up", 0.78, "one wrist is above its shoulder")

        if len(self.history) >= 3:
            motion = self._wrist_motion(self.history)
            if motion > 16:
                return ActionResult("waving_or_swinging", min(0.95, motion / 45), "wrist motion is large")

        lean = self._torso_lean_degrees(keypoints)
        if abs(lean) > 16:
            label = "lean_left" if lean < 0 else "lean_right"
            return ActionResult(label, min(0.9, abs(lean) / 35), f"torso lean is {lean:.1f} degrees")

        return ActionResult("standing", 0.7, "stable upright pose")

    def _visible_points(self, keypoints: np.ndarray) -> int:
        if keypoints.shape[1] < 3:
            return int(keypoints.shape[0])
        return int(np.count_nonzero(keypoints[:, 2] >= self.min_keypoint_conf))

    def _point_visible(self, keypoints: np.ndarray, idx: int) -> bool:
        return keypoints.shape[1] < 3 or keypoints[idx, 2] >= self.min_keypoint_conf

    def _both_wrists_above_shoulders(self, keypoints: np.ndarray) -> bool:
        pairs = [
            (COCO_LEFT_WRIST, COCO_LEFT_SHOULDER),
            (COCO_RIGHT_WRIST, COCO_RIGHT_SHOULDER),
        ]
        return all(
            self._point_visible(keypoints, wrist)
            and self._point_visible(keypoints, shoulder)
            and keypoints[wrist, 1] < keypoints[shoulder, 1]
            for wrist, shoulder in pairs
        )

    def _single_arm_raised(self, keypoints: np.ndarray) -> bool:
        for wrist, shoulder in [
            (COCO_LEFT_WRIST, COCO_LEFT_SHOULDER),
            (COCO_RIGHT_WRIST, COCO_RIGHT_SHOULDER),
        ]:
            if (
                self._point_visible(keypoints, wrist)
                and self._point_visible(keypoints, shoulder)
                and keypoints[wrist, 1] < keypoints[shoulder, 1]
            ):
                return True
        return False

    def _wrist_motion(self, history: Iterable[np.ndarray]) -> float:
        frames = list(history)
        motions = []
        for wrist in [COCO_LEFT_WRIST, COCO_RIGHT_WRIST]:
            points = [frame[wrist, :2] for frame in frames if self._point_visible(frame, wrist)]
            if len(points) >= 2:
                diffs = [np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points))]
                motions.append(float(np.mean(diffs)))
        return max(motions, default=0.0)

    def _torso_lean_degrees(self, keypoints: np.ndarray) -> float:
        required = [COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER, COCO_LEFT_HIP, COCO_RIGHT_HIP]
        if not all(self._point_visible(keypoints, idx) for idx in required):
            return 0.0

        shoulder_mid = (keypoints[COCO_LEFT_SHOULDER, :2] + keypoints[COCO_RIGHT_SHOULDER, :2]) / 2
        hip_mid = (keypoints[COCO_LEFT_HIP, :2] + keypoints[COCO_RIGHT_HIP, :2]) / 2
        dx, dy = shoulder_mid - hip_mid
        return degrees(atan2(dx, -dy))
