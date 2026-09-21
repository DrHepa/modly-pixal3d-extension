"""Decode-only video metadata probe for the isolated scene-preparation lane."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import cv2


def probe(path: Path) -> dict:
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise ValueError("Scene video cannot be decoded")
    width = height = count = 0
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            current_height, current_width = frame.shape[:2]
            if count == 0:
                width, height = current_width, current_height
            elif (current_width, current_height) != (width, height):
                raise ValueError("Scene video frames must have consistent dimensions")
            count += 1
            if count > 100_000:
                raise ValueError("Scene video exceeds the 100000-frame safety limit")
    finally:
        video.release()
    if width <= 0 or height <= 0 or count < 2:
        raise ValueError("Scene video must contain at least two decodable frames")
    return {"width": width, "height": height, "frameCount": count}


def main() -> None:
    try:
        if len(sys.argv) != 2:
            raise ValueError("Expected one staged video path")
        source = Path(sys.argv[1]).resolve(strict=True)
        result = probe(source)
        output = source.parent / "video-metadata.json"
        output.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        print(json.dumps({"type": "done", "metadata_path": str(output)}, sort_keys=True), flush=True)
    except Exception as exc:
        print(json.dumps({"type": "error", "message": str(exc)}, sort_keys=True), flush=True)
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
