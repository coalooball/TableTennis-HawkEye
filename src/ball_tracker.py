from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BallDetection:
    center: tuple[int, int] | None
    confidence: float
    accepted: bool
    radius: float = 0.0


class BallTracker:
    def __init__(
        self,
        history_size: int = 40,
        max_missing_frames: int = 8,
        max_jump: float = 180.0,
    ):
        self.trajectory: deque[tuple[int, int]] = deque(maxlen=history_size)
        self.max_missing_frames = max_missing_frames
        self.max_jump = max_jump
        self.missing_frames = 0
        self.last_detection: BallDetection | None = None

    def clear(self) -> None:
        self.trajectory.clear()
        self.missing_frames = 0
        self.last_detection = None

    def update(self, detection: BallDetection) -> tuple[int, int] | None:
        if detection.accepted and detection.center is not None and not self._is_outlier(detection.center):
            self.trajectory.append(detection.center)
            self.missing_frames = 0
            self.last_detection = detection
            return detection.center

        self.missing_frames += 1
        self.last_detection = BallDetection(None, 0.0, False)
        if self.missing_frames > self.max_missing_frames:
            self.trajectory.clear()
        return None

    def predict_next(self) -> tuple[int, int] | None:
        if len(self.trajectory) < 2:
            return self.trajectory[-1] if self.trajectory else None
        x1, y1 = self.trajectory[-2]
        x2, y2 = self.trajectory[-1]
        return (x2 + (x2 - x1), y2 + (y2 - y1))

    def draw(self, frame: np.ndarray, show_trajectory: bool = True) -> None:
        points = list(self.trajectory)
        if show_trajectory and len(points) >= 2:
            for index in range(1, len(points)):
                thickness = 1 + index // 12
                cv2.line(frame, points[index - 1], points[index], (40, 220, 255), thickness, cv2.LINE_AA)

        if points:
            cv2.circle(frame, points[-1], 7, (0, 80, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, points[-1], 2, (0, 80, 255), -1, cv2.LINE_AA)

    def _is_outlier(self, center: tuple[int, int]) -> bool:
        if not self.trajectory:
            return False
        reference = self.predict_next() or self.trajectory[-1]
        distance = np.hypot(center[0] - reference[0], center[1] - reference[1])
        allowed_jump = self.max_jump + min(self.missing_frames, self.max_missing_frames) * 22
        return distance > allowed_jump
