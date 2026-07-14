"""
deploy.py

Production Deployment Script

Author: Argha Sarkar Project
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


class DeploymentPipeline:
    """
    Production Deployment Pipeline.
    """

    def __init__(self, args):

        self.args = args

        self.root = Path.cwd()

        self.deploy_dir = self.root / "deployment"

        self.deploy_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ---------------------------------------------------------

    def verify_files(self):

        print("=" * 70)
        print("VERIFYING DEPLOYMENT FILES")
        print("=" * 70)

        required = [
            self.args.model,
            "Dockerfile",
            "docker-compose.yml",
            "requirements.txt",
        ]

        missing = []

        for file in required:

            if not Path(file).exists():

                missing.append(file)

        if len(missing):

            print()

            print("Missing Files")

            for item in missing:

                print(item)

            raise FileNotFoundError

        print("Verification Successful")

    # ---------------------------------------------------------

    def copy_files(self):

        print("=" * 70)
        print("COPYING FILES")
        print("=" * 70)

        folders = [
            "src",
            "configs",
            "models",
            "scripts",
        ]

        for folder in folders:

            folder = Path(folder)

            if folder.exists():

                shutil.copytree(
                    folder,
                    self.deploy_dir / folder.name,
                    dirs_exist_ok=True,
                )

        files = [
            "Dockerfile",
            "docker-compose.yml",
            "requirements.txt",
            "requirements-api.txt",
            "README.md",
        ]

        for file in files:

            file = Path(file)

            if file.exists():

                shutil.copy2(
                    file,
                    self.deploy_dir,
                )

    # ---------------------------------------------------------

    def build_image(self):

        print("=" * 70)
        print("BUILDING DOCKER IMAGE")
        print("=" * 70)

        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                self.args.image,
                ".",
            ],
            check=True,
        )

    # ---------------------------------------------------------

    def run_container(self):

        print("=" * 70)
        print("RUNNING CONTAINER")
        print("=" * 70)

        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.args.container,
                "-p",
                f"{self.args.port}:8000",
                self.args.image,
            ],
            check=True,
        )

    # ---------------------------------------------------------

    def compose_up(self):

        compose = Path("docker-compose.yml")

        if not compose.exists():

            return

        print("=" * 70)
        print("DOCKER COMPOSE")
        print("=" * 70)

        subprocess.run(
            [
                "docker",
                "compose",
                "up",
                "-d",
            ],
            check=True,
        )

    # ---------------------------------------------------------

    def export_archive(self):

        print("=" * 70)
        print("CREATING DEPLOYMENT PACKAGE")
        print("=" * 70)

        archive = shutil.make_archive(
            base_name="ASL-Vision-Deployment",
            format="zip",
            root_dir=self.deploy_dir,
        )

        print()

        print("Package Created")

        print(archive)

    # ---------------------------------------------------------

    def health_check(self):

        import time

        import requests

        print("=" * 70)
        print("HEALTH CHECK")
        print("=" * 70)

        url = f"http://127.0.0.1:{self.args.port}" "/api/v1/health"

        for _ in range(20):

            try:

                response = requests.get(
                    url,
                    timeout=5,
                )

                if response.status_code == 200:

                    print("API Healthy")

                    return

            except Exception:

                pass

            time.sleep(2)

        print("Health Check Failed")

    # ---------------------------------------------------------

    def deploy(self):

        self.verify_files()

        self.copy_files()

        self.build_image()

        self.run_container()

        self.health_check()

        self.export_archive()

        print()

        print("=" * 70)
        print("DEPLOYMENT SUCCESSFUL")
        print("=" * 70)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------


def arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="models/best_model.keras",
    )

    parser.add_argument(
        "--image",
        default="asl-vision:latest",
    )

    parser.add_argument(
        "--container",
        default="asl-vision",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    args = arguments()

    pipeline = DeploymentPipeline(args)

    pipeline.deploy()
