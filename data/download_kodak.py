"""Download the Kodak PhotoCD dataset (24 images, ~12 MB total)."""

import urllib.request
from pathlib import Path


def main():
    out_dir = Path(__file__).parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = "http://r0k.us/graphics/kodak/kodak/"

    for i in range(1, 25):
        filename = f"kodim{i:02d}.png"
        dest = out_dir / filename
        if dest.exists():
            print(f"  skip {filename} (exists)")
            continue
        url = base_url + filename
        print(f"  downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)

    print(f"\nDone — {len(list(out_dir.glob('*.png')))} images in {out_dir}")


if __name__ == "__main__":
    main()
