"""
build.py

Production Build Script

Author: Argha Sarkar Project
"""

from pathlib import Path
import shutil
import subprocess
import sys


class ProjectBuilder:

    def __init__(self):

        self.root = Path.cwd()

        self.build_dir = self.root / "build"

        self.dist_dir = self.root / "dist"

    # ---------------------------------------------------------

    def clean(self):

        print("=" * 70)
        print("Cleaning Build Directories")
        print("=" * 70)

        shutil.rmtree(
            self.build_dir,
            ignore_errors=True,
        )

        shutil.rmtree(
            self.dist_dir,
            ignore_errors=True,
        )

        self.build_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.dist_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ---------------------------------------------------------

    def install_dependencies(self):

        print("=" * 70)
        print("Installing Dependencies")
        print("=" * 70)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                "requirements.txt",
            ],
            check=True,
        )

        api_requirements = Path(
            "requirements-api.txt"
        )

        if api_requirements.exists():

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    "requirements-api.txt",
                ],
                check=True,
            )

    # ---------------------------------------------------------

    def create_build(self):

        print("=" * 70)
        print("Creating Build")
        print("=" * 70)

        folders = [

            "src",

            "configs",

            "models",

            "reports",

            "scripts",

        ]

        for folder in folders:

            folder = Path(folder)

            if folder.exists():

                destination = (
                    self.build_dir
                    / folder.name
                )

                shutil.copytree(
                    folder,
                    destination,
                    dirs_exist_ok=True,
                )

        files = [

            "README.md",

            "requirements.txt",

            "requirements-api.txt",

            "Dockerfile",

            "docker-compose.yml",

        ]

        for file in files:

            file = Path(file)

            if file.exists():

                shutil.copy2(
                    file,
                    self.build_dir,
                )

    # ---------------------------------------------------------

    def compress(self):

        print("=" * 70)
        print("Compressing Build")
        print("=" * 70)

        archive = shutil.make_archive(

            base_name=str(
                self.dist_dir
                / "ASL-Vision"
            ),

            format="zip",

            root_dir=self.build_dir,

        )

        print()

        print("Created:")

        print(archive)

        return archive

    # ---------------------------------------------------------

    def verify(self):

        print("=" * 70)
        print("Verifying Build")
        print("=" * 70)

        files = list(
            self.build_dir.rglob("*")
        )

        print(
            f"Files : {len(files)}"
        )

        return len(files) > 0

    # ---------------------------------------------------------

    def build(self):

        self.clean()

        self.install_dependencies()

        self.create_build()

        self.verify()

        archive = self.compress()

        print()

        print("=" * 70)
        print("BUILD SUCCESSFUL")
        print("=" * 70)

        return archive


if __name__ == "__main__":

    builder = ProjectBuilder()

    builder.build()