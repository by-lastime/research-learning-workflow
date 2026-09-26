#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
from rapidocr_onnxruntime import RapidOCR


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: ocr_images.py image...")

    engine = RapidOCR()
    for value in sys.argv[1:]:
        path = Path(value)
        print(f"\n===== {path.name} =====\n")
        with Image.open(path) as image:
            image = image.convert("RGB")
            # Very tall mobile screenshots are divided into strips so the OCR
            # detector retains enough resolution for small Chinese characters.
            strip_height = 1800
            for top in range(0, image.height, strip_height):
                crop = image.crop((0, top, image.width, min(top + strip_height, image.height)))
                result, _ = engine(crop)
                if not result:
                    continue
                result.sort(key=lambda item: (min(point[1] for point in item[0]), min(point[0] for point in item[0])))
                for _, text, score in result:
                    if score >= 0.45:
                        print(text)


if __name__ == "__main__":
    main()
