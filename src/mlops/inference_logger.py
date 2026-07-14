"""
inference_logger.py

Production Inference Logger

Author: Argha Sarkar Project
"""

import json
import uuid
from datetime import datetime
from pathlib import Path


class InferenceLogger:
    """
    Production inference logger.
    """

    def __init__(
        self,
        log_directory="logs/inference",
    ):

        self.log_directory = Path(log_directory)

        self.log_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ---------------------------------------------------------

    def _timestamp(self):

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------------------------------------------------------

    def _generate_id(self):

        return str(uuid.uuid4())

    # ---------------------------------------------------------

    def log_prediction(
        self,
        prediction,
        confidence,
        processing_time,
        filename=None,
        probabilities=None,
    ):

        record = {
            "request_id": self._generate_id(),
            "timestamp": self._timestamp(),
            "filename": filename,
            "prediction": prediction,
            "confidence": float(confidence),
            "processing_time_ms": float(processing_time),
            "probabilities": probabilities,
        }

        logfile = self.log_directory / f"{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(
            logfile,
            "a",
            encoding="utf-8",
        ) as file:

            file.write(json.dumps(record))

            file.write("\n")

        return record

    # ---------------------------------------------------------

    def log_batch(
        self,
        predictions,
    ):

        for prediction in predictions:

            self.log_prediction(
                prediction=prediction.get("prediction"),
                confidence=prediction.get("confidence"),
                processing_time=prediction.get(
                    "processing_time_ms",
                    0,
                ),
                filename=prediction.get("filename"),
                probabilities=prediction.get("probabilities"),
            )

    # ---------------------------------------------------------

    def load_logs(
        self,
        logfile,
    ):

        logfile = Path(logfile)

        if not logfile.exists():

            return []

        records = []

        with open(
            logfile,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                records.append(json.loads(line))

        return records

    # ---------------------------------------------------------

    def summary(
        self,
        logfile,
    ):

        logs = self.load_logs(logfile)

        if len(logs) == 0:

            return {}

        confidences = [item["confidence"] for item in logs]

        processing = [item["processing_time_ms"] for item in logs]

        summary = {
            "total_predictions": len(logs),
            "average_confidence": sum(confidences) / len(confidences),
            "minimum_confidence": min(confidences),
            "maximum_confidence": max(confidences),
            "average_processing_time_ms": sum(processing) / len(processing),
        }

        return summary

    # ---------------------------------------------------------

    def save_summary(
        self,
        logfile,
    ):

        summary = self.summary(logfile)

        output = self.log_directory / "summary.json"

        with open(
            output,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                indent=4,
            )

        return output

    # ---------------------------------------------------------

    def clear_logs(self):

        for file in self.log_directory.glob("*.jsonl"):

            file.unlink()

    # ---------------------------------------------------------

    def print_summary(
        self,
        logfile,
    ):

        summary = self.summary(logfile)

        print("\n" + "=" * 70)

        print("INFERENCE SUMMARY")

        print("=" * 70)

        for key, value in summary.items():

            print(f"{key:<30}: {value}")


if __name__ == "__main__":

    logger = InferenceLogger()

    logger.log_prediction(
        prediction="A",
        confidence=0.998,
        processing_time=12.4,
        filename="sample.png",
    )
