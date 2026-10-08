import math
import time


class RiskAssessor:
    """
    Contextual risk assessment module.

    This module matches the presentation pipeline:
    Perception -> Behavioral Analysis -> Contextual Risk Assessment -> Decision Making.

    In the real system, YOLO would provide person detection confidence.
    For this demo, the confidence is simulated/estimated from the intruder position.
    """

    def __init__(self, restricted_zone=(0.0, -2.8), restricted_radius=1.4):
        self.restricted_zone = restricted_zone
        self.restricted_radius = restricted_radius
        self.first_seen_time = None
        self.last_position = None
        self.start_time = time.time()

    def distance(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def update_dwell_time(self, person_detected):
        if person_detected:
            if self.first_seen_time is None:
                self.first_seen_time = time.time()
            return time.time() - self.first_seen_time

        self.first_seen_time = None
        return 0.0

    def estimate_yolo_confidence(self, intruder_pos):
        """
        Demo substitute for YOLOv8n confidence.

        In a camera-based system, this value would come from:
        model(frame)[0].boxes.conf
        """
        if intruder_pos is None:
            return 0.0

        dist_to_building = self.distance(intruder_pos, (0.0, 0.0))

        if dist_to_building <= 2.5:
            return 0.90
        if dist_to_building <= 4.5:
            return 0.75
        return 0.55

    def assess(self, intruder_pos, is_night=False):
        if intruder_pos is None:
            return {
                "score": 0.0,
                "level": "LOW",
                "dwell_time": 0.0,
                "distance_to_restricted": 999.0,
                "yolo_confidence": 0.0,
            }

        person_detected = True
        dwell_time = self.update_dwell_time(person_detected)
        distance_to_restricted = self.distance(intruder_pos, self.restricted_zone)
        yolo_confidence = self.estimate_yolo_confidence(intruder_pos)

        time_score = 1.0 if is_night else 0.3
        restricted_score = 1.0 if distance_to_restricted <= self.restricted_radius else 0.0
        dwell_score = min(dwell_time / 8.0, 1.0)
        distance_score = max(0.0, 1.0 - distance_to_restricted / 7.0)
        confidence_score = yolo_confidence

        risk_score = (
            0.15 * time_score
            + 0.30 * restricted_score
            + 0.20 * dwell_score
            + 0.20 * distance_score
            + 0.15 * confidence_score
        )

        if risk_score >= 0.70:
            level = "HIGH"
        elif risk_score >= 0.40:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "score": risk_score,
            "level": level,
            "dwell_time": dwell_time,
            "distance_to_restricted": distance_to_restricted,
            "yolo_confidence": yolo_confidence,
        }
