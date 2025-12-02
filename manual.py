import os
import argparse
import threading
from pathlib import Path
from typing import Tuple, List

from PIL import Image, ImageSequence
import numpy as np
import cv2
from paddleocr import PaddleOCR
from pdf2image import convert_from_path

# Thread-safety locks
_ocr_lock = threading.Lock()
_init_lock = threading.Lock()

_ocr = None
_initialized = False

SUPPORTED_EXTENSIONS = {'.tif', '.tiff', '.pdf'}


def _init_ocr() -> PaddleOCR:
    global _ocr, _initialized
    if _initialized and _ocr is not None:
        return _ocr
    with _init_lock:
        if not _initialized or _ocr is None:
            _ocr = PaddleOCR(lang='en')
            _initialized = True
    return _ocr


def _img_to_bgr_array(img: Image.Image) -> np.ndarray:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def _preprocess_for_ocr(img: Image.Image) -> np.ndarray:
    max_dim = max(img.size)
    if max_dim < 1500:
        scale = min(2000 / max_dim, 2.0)
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    bgr = _img_to_bgr_array(img)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.std() < 25:
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 15
        )
        bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    return bgr


def _extract_text_from_result(result) -> List[str]:
    texts = []
    for res in result:
        if hasattr(res, 'json') and res.json:
            json_data = res.json
            if 'res' in json_data and 'rec_texts' in json_data['res']:
                for text in json_data['res']['rec_texts']:
                    if text:
                        texts.append(text)
    return texts


def _ocr_images(images: List[Image.Image]) -> str:
    ocr = _init_ocr()
    texts = []

    for img in images:
        bgr = _preprocess_for_ocr(img)
        with _ocr_lock:
            result = ocr.predict(bgr)
        page_texts = _extract_text_from_result(result)
        texts.extend(page_texts)

    return "\n\n".join(t for t in texts if t).strip()


def ocr_tiff(filepath: str) -> str:
    images = []
    with Image.open(filepath) as im:
        for frame in ImageSequence.Iterator(im):
            images.append(frame.copy())
    return _ocr_images(images)


def ocr_pdf(filepath: str) -> str:
    images = convert_from_path(filepath)
    return _ocr_images(images)


def process_file(filepath: Path) -> Tuple[str, str]:
    ext = filepath.suffix.lower()
    if ext in {'.tif', '.tiff'}:
        text = ocr_tiff(str(filepath))
    elif ext == '.pdf':
        text = ocr_pdf(str(filepath))
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return text


def find_files(directory: Path) -> List[Path]:
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(
        description='OCR files in a directory and save results as .txt files'
    )
    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing .tif, .tiff, or .pdf files to OCR'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for .txt files (default: same as input)'
    )
    args = parser.parse_args()

    input_dir = Path(args.directory)
    if not input_dir.is_dir():
        print(f"Error: {args.directory} is not a valid directory")
        return 1

    output_dir = Path(args.output) if args.output else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_files(input_dir)
    if not files:
        print(f"No .tif, .tiff, or .pdf files found in {input_dir}")
        return 0

    print(f"Found {len(files)} file(s) to process")

    # Initialize OCR once before processing
    print("Initializing PaddleOCR...")
    _init_ocr()

    for filepath in files:
        print(f"Processing: {filepath.name}")
        try:
            text = process_file(filepath)
            output_file = output_dir / f"{filepath.stem}.txt"
            output_file.write_text(text, encoding='utf-8')
            print(f"  -> Saved: {output_file.name}")
        except Exception as e:
            print(f"  -> Error: {e}")

    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
