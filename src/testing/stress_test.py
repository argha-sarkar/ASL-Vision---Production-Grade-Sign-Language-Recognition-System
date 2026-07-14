"""
stress_test.py

Production API Stress Test

Author: Argha Sarkar Project
"""

import concurrent.futures
import statistics
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests


class StressTester:
    """
    Stress testing utility for FastAPI deployment.
    """

    def __init__(
        self,
        api_url="http://127.0.0.1:8000/api/v1/predict",
        image_size=(224, 224),
        output_dir="reports/testing",
    ):

        self.api_url = api_url

        self.image_size = image_size

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(

            parents=True,

            exist_ok=True,

        )

        self.lock = threading.Lock()

        self.results = []

    # ---------------------------------------------------------

    def create_image(self):

        image = np.random.randint(

            0,

            255,

            (

                self.image_size[0],

                self.image_size[1],

                3,

            ),

            dtype=np.uint8,

        )

        _, encoded = cv2.imencode(

            ".png",

            image,

        )

        return encoded.tobytes()

    # ---------------------------------------------------------

    def send_request(self):

        image = self.create_image()

        files = {

            "file": (

                "sample.png",

                image,

                "image/png",

            )

        }

        start = time.perf_counter()

        try:

            response = requests.post(

                self.api_url,

                files=files,

                timeout=30,

            )

            latency = (

                time.perf_counter()

                - start

            ) * 1000

            success = response.status_code == 200

        except Exception:

            latency = (

                time.perf_counter()

                - start

            ) * 1000

            success = False

        with self.lock:

            self.results.append(

                {

                    "latency": latency,

                    "success": success,

                }

            )

    # ---------------------------------------------------------

    def run(

        self,

        total_requests=500,

        workers=20,

    ):

        self.results = []

        print()

        print("=" * 70)

        print("STARTING STRESS TEST")

        print("=" * 70)

        overall_start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(

            max_workers=workers

        ) as executor:

            futures = [

                executor.submit(

                    self.send_request

                )

                for _ in range(

                    total_requests

                )

            ]

            concurrent.futures.wait(

                futures

            )

        overall_end = time.perf_counter()

        report = self.summary(

            overall_end

            - overall_start

        )

        self.save(report)

        self.print(report)

        return report

    # ---------------------------------------------------------

    def summary(

        self,

        elapsed,

    ):

        latencies = [

            item["latency"]

            for item in self.results

        ]

        success = sum(

            item["success"]

            for item in self.results

        )

        failed = len(

            self.results

        ) - success

        throughput = (

            len(self.results)

            / elapsed

        )

        report = {

            "total_requests":

                len(self.results),

            "successful":

                success,

            "failed":

                failed,

            "success_rate":

                (

                    success

                    / len(self.results)

                )

                * 100,

            "average_latency_ms":

                statistics.mean(

                    latencies

                ),

            "minimum_latency_ms":

                min(

                    latencies

                ),

            "maximum_latency_ms":

                max(

                    latencies

                ),

            "median_latency_ms":

                statistics.median(

                    latencies

                ),

            "throughput_rps":

                throughput,

            "test_duration_sec":

                elapsed,

        }

        return report

    # ---------------------------------------------------------

    def save(

        self,

        report,

    ):

        import pandas as pd

        dataframe = pd.DataFrame(

            report.items(),

            columns=[

                "Metric",

                "Value",

            ],

        )

        dataframe.to_csv(

            self.output_dir

            / "stress_test.csv",

            index=False,

        )

    # ---------------------------------------------------------

    def print(

        self,

        report,

    ):

        print()

        print("=" * 70)

        print("STRESS TEST REPORT")

        print("=" * 70)

        for key, value in report.items():

            if isinstance(

                value,

                float,

            ):

                print(

                    f"{key:<30}: {value:.4f}"

                )

            else:

                print(

                    f"{key:<30}: {value}"

                )

        print("=" * 70)


if __name__ == "__main__":

    tester = StressTester(

        api_url="http://127.0.0.1:8000/api/v1/predict"

    )

    tester.run(

        total_requests=500,

        workers=20,

    )