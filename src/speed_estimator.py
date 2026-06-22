from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from table_calibration import TableCalibrator


class BallSpeedEstimator:
    def __init__(self, window_size: int = 6, table_diagonal_meters: float = 3.07):
        self.samples: deque[tuple[int, tuple[int, int], float]] = deque(maxlen=window_size)
        self.table_diagonal_meters = table_diagonal_meters
        self.last_speed_px_s: float | None = None
        self.last_speed_m_s: float | None = None

    def clear(self) -> None:
        self.samples.clear()
        self.last_speed_px_s = None
        self.last_speed_m_s = None

    def update(
        self,
        frame_index: int,
        point: tuple[int, int] | None,
        fps: float,
        table: TableCalibrator,
    ) -> tuple[float | None, float | None]:
        if point is None or fps <= 0:
            return self.last_speed_px_s, self.last_speed_m_s

        self.samples.append((frame_index, point, frame_index / fps))
        if len(self.samples) < 2:
            return self.last_speed_px_s, self.last_speed_m_s

        start_frame, start_point, start_time = self.samples[0]
        end_frame, end_point, end_time = self.samples[-1]
        if end_frame == start_frame or end_time <= start_time:
            return self.last_speed_px_s, self.last_speed_m_s

        distance_px = float(np.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1]))
        self.last_speed_px_s = distance_px / (end_time - start_time)
        meters_per_pixel = self._meters_per_pixel(table)
        self.last_speed_m_s = self.last_speed_px_s * meters_per_pixel if meters_per_pixel is not None else None
        return self.last_speed_px_s, self.last_speed_m_s

    def draw(self, frame) -> None:
        if self.last_speed_px_s is None:
            return

        if self.last_speed_m_s is None:
            text = f"Ball speed: {self.last_speed_px_s:.0f} px/s"
        else:
            text = f"Ball speed: {self.last_speed_m_s:.2f} m/s  {self.last_speed_m_s * 3.6:.1f} km/h"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.72
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        padding = 8
        left = 0
        top = 0
        right = min(frame.shape[1] - 1, text_width + padding * 2)
        bottom = min(frame.shape[0] - 1, text_height + baseline + padding * 2)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), thickness=-1)
        cv2.putText(
            frame,
            text,
            (left + padding, top + padding + text_height),
            font,
            font_scale,
            (80, 220, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _meters_per_pixel(self, table: TableCalibrator) -> float | None:
        if not table.is_complete:
            return None
        top_left, _, bottom_right, _ = table.points
        diagonal_pixels = float(np.hypot(bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))
        if diagonal_pixels <= 0:
            return None
        return self.table_diagonal_meters / diagonal_pixels
