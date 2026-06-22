from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from ball_tracker import BallDetection


DEFAULT_BALL_MODEL = Path("models/yolo11s-ball.pt")
SPORTS_BALL_CLASS_ID = 32


class YoloBallDetector:
    def __init__(self, model_path: str | Path = DEFAULT_BALL_MODEL, conf: float = 0.18):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Ball model not found: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.conf = conf

    def detect(self, frame: np.ndarray, predicted: tuple[int, int] | None = None) -> BallDetection:
        results = self.model.predict(frame, conf=self.conf, classes=[SPORTS_BALL_CLASS_ID], verbose=False)
        result = results[0]
        if result.boxes is None or result.boxes.xyxy is None or len(result.boxes.xyxy) == 0:
            return BallDetection(None, 0.0, False)

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes))
        candidates: list[tuple[float, tuple[int, int], float, float, tuple[int, int, int, int]]] = []

        for box, confidence in zip(boxes, confs):
            x1, y1, x2, y2 = box
            width = max(float(x2 - x1), 1.0)
            height = max(float(y2 - y1), 1.0)
            center = (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))
            radius = max(width, height) / 2
            detection_box = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
            score = float(confidence)
            if predicted is not None:
                distance = np.hypot(center[0] - predicted[0], center[1] - predicted[1])
                score -= min(distance / 320.0, 1.0) * 0.25
            candidates.append((score, center, float(confidence), radius, detection_box))

        score, center, confidence, radius, detection_box = max(candidates, key=lambda item: item[0])
        return BallDetection(center, confidence, score >= self.conf * 0.5, radius, detection_box)
