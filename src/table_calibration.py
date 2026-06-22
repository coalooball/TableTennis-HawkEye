from __future__ import annotations

import cv2
import numpy as np


TABLE_POINT_LABELS = ("TL", "TR", "BR", "BL")
TABLE_POINT_PROMPTS = (
    "Click table top-left corner",
    "Click table top-right corner",
    "Click table bottom-right corner",
    "Click table bottom-left corner",
)


class TableCalibrator:
    def __init__(self):
        self.points: list[tuple[int, int]] = []

    @property
    def is_complete(self) -> bool:
        return len(self.points) == 4

    def reset(self) -> None:
        self.points.clear()

    def add_point(self, point: tuple[int, int]) -> None:
        if self.is_complete:
            return
        self.points.append((int(point[0]), int(point[1])))

    def next_prompt(self) -> str:
        if self.is_complete:
            return "Table calibrated"
        return TABLE_POINT_PROMPTS[len(self.points)]

    def contains(self, point: tuple[int, int]) -> bool:
        if not self.is_complete:
            return False
        polygon = np.array(self.points, dtype=np.float32)
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

    def draw(self, frame: np.ndarray) -> None:
        if not self.points:
            return

        for index, point in enumerate(self.points):
            cv2.circle(frame, point, 6, (0, 210, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                TABLE_POINT_LABELS[index],
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 210, 255),
                2,
                cv2.LINE_AA,
            )

        pts = np.array(self.points, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(frame, [pts], self.is_complete, (0, 210, 255), 2, cv2.LINE_AA)

        if not self.is_complete:
            cv2.rectangle(frame, (12, 70), (390, 112), (0, 0, 0), thickness=-1)
            cv2.putText(
                frame,
                self.next_prompt(),
                (24, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 210, 255),
                2,
                cv2.LINE_AA,
            )
