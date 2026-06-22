from __future__ import annotations

import cv2
import numpy as np

from table_calibration import TableCalibrator


class LandingDetector:
    def __init__(self, min_vertical_speed: float = 3.0, cooldown_frames: int = 8):
        self.points: list[tuple[int, int]] = []
        self.min_vertical_speed = min_vertical_speed
        self.cooldown_frames = cooldown_frames
        self._cooldown = 0

    def clear(self) -> None:
        self.points.clear()
        self._cooldown = 0

    def update(self, trajectory: list[tuple[int, int]], table: TableCalibrator) -> tuple[int, int] | None:
        if self._cooldown > 0:
            self._cooldown -= 1
            return None
        if not table.is_complete or len(trajectory) < 5:
            return None

        p0, p1, p2, p3, p4 = trajectory[-5:]
        prev_velocity = ((p2[1] - p0[1]) + (p3[1] - p1[1])) / 2.0
        next_velocity = p4[1] - p3[1]
        candidate = p3

        if prev_velocity <= self.min_vertical_speed or next_velocity >= -self.min_vertical_speed:
            return None
        if not table.contains(candidate):
            return None
        if self.points and np.hypot(candidate[0] - self.points[-1][0], candidate[1] - self.points[-1][1]) < 16:
            return None

        self.points.append(candidate)
        self._cooldown = self.cooldown_frames
        return candidate

    def draw(self, frame) -> None:
        for index, point in enumerate(self.points, start=1):
            cv2.circle(frame, point, 12, (40, 255, 80), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                str(index),
                (point[0] + 14, point[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (40, 255, 80),
                2,
                cv2.LINE_AA,
            )
