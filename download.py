"""Download the Wonders of the World image dataset into ./data."""
import shutil
from pathlib import Path

import kagglehub

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    src = Path(kagglehub.dataset_download("balabaskar/wonders-of-the-world-image-classification"))
    print(f"Downloaded to cache: {src}")
    shutil.copytree(src, DATA_DIR, dirs_exist_ok=True)
    count = sum(1 for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    print(f"Copied {count} images into {DATA_DIR}")


if __name__ == "__main__":
    main()