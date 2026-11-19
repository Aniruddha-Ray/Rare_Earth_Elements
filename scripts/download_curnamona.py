"""Download the northern Curnamona Province REE dataset (DEEP-SEAM v1.0) from Zenodo.

Source : Luo, Farahbakhsh, Hore & Muller (2026) DEEP-SEAM, Geosci. Model Dev. 19, 2593-2625.
Record : https://doi.org/10.5281/zenodo.17098677 (GPL-3.0; underlying layers from SARIG, CC-BY 4.0)

    python scripts/download_curnamona.py            # -> data/external/curnamona/
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/api/records/17098677/files/EarthByte/MPM_Curnamona_REE-DEEP-SEAM.zip/content"
MD5 = "558b9d7898fe1b849bbe6906d09c773b"


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/external")
    args = ap.parse_args()
    out = Path(args.out)
    zip_path = out / "curnamona_zip" / "MPM_Curnamona_REE-DEEP-SEAM.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists() or md5sum(zip_path) != MD5:
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, zip_path)
    digest = md5sum(zip_path)
    if digest != MD5:
        raise SystemExit(f"MD5 mismatch: got {digest}, expected {MD5}")
    print("Checksum OK")
    dest = out / "curnamona"
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                raise SystemExit(f"Refusing unsafe path in archive: {name}")
        z.extractall(dest)
    print(f"Extracted to {dest}")


if __name__ == "__main__":
    main()
