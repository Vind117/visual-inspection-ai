import cv2
import numpy as np
from ultralytics import YOLO
import time

class DefectDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25):
        start_time = time.time()
        results = self.model(image, conf=conf_threshold)[0]
        inference_time = (time.time() - start_time) * 1000  # Latency in ms

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        names = results.names

        annotated_img = results.plot()

        detections = []
        for box, score, cls in zip(boxes, scores, classes):
            detections.append({
                "Defect Class": names[int(cls)],
                "Confidence": f"{float(score):.2%}",
                "BBox [X1, Y1, X2, Y2]": [int(b) for b in box]
            })

        return annotated_img, detections, inference_time