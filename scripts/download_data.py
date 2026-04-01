import subprocess
import zipfile
from pathlib import Path
import sys
from kaggle.api.kaggle_api_extended import KaggleApi


DATASET = "ARYA SHAH/Highway Traffic Videos Dataset"

data_dir = Path("data")

data_dir.mkdir(parents=True, exist_ok=True)


def download_from_kaggle(data_dir: Path):

    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading Avazu competition files into: {data_dir}")

    api.competition_download_files(
        competition="avazu-ctr-prediction",
        path=str(data_dir),
        quiet=False,
        force=False,
    )

    print("Files after download:", [p.name for p in data_dir.iterdir()])


def unzip_files(data_dir: Path):
    """Extract downloaded zip files."""
    for zip_file in data_dir.glob("*.zip"):
        print(f"Extracting {zip_file.name}")
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(data_dir)

        zip_file.unlink()


def main():
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    download_from_kaggle(raw_dir)
    unzip_files(raw_dir)

    print("Contents:", list(raw_dir.iterdir()))

    print("Dataset ready in:", raw_dir.resolve())


if __name__ == "__main__":
    main()
