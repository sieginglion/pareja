#!/usr/bin/env python3
"""
Render each page of a PDF to an image, summarize each page with OpenAI, and
concatenate the summaries into a text output.
"""

from __future__ import annotations

import base64
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI

try:
    import fitz
except ImportError as exc:  # pragma: no cover - import guard for runtime setup
    raise SystemExit(
        "PyMuPDF is required. Install dependencies from requirements.txt first."
    ) from exc

load_dotenv()

MODEL = "gpt-5.4"
PROMPT = "Summarize the page in detail. Ignore tables."
# Render at 150 DPI, then encode as medium-quality JPEG to keep text readable
# while reducing payload size versus lossless page images.
DPI = 150
JPEG_QUALITY = 80
CONCURRENCY = 4


def render_pdf_pages(pdf_path: Path, dpi: int) -> Iterable[bytes]:
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as document:
        for page in document:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            yield pixmap.tobytes("jpeg", jpg_quality=JPEG_QUALITY)


def jpeg_bytes_to_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def summarize_page(
    client: OpenAI,
    model: str,
    prompt: str,
    image_bytes: bytes,
) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": jpeg_bytes_to_data_url(image_bytes),
                        "detail": "original",
                    },
                ],
            }
        ],
        reasoning={"effort": "high"},
    )
    return (getattr(response, "output_text", "") or "").strip()


def summarize_batch(
    client: OpenAI,
    batch: list[tuple[int, bytes]],
) -> list[tuple[int, str]]:
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [
            executor.submit(
                summarize_page,
                client,
                MODEL,
                PROMPT,
                image_bytes,
            )
            for _, image_bytes in batch
        ]
        summaries = [future.result() for future in futures]
    return [
        (page_number, summary)
        for (page_number, _), summary in zip(batch, summaries, strict=True)
    ]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} <pdf>")

    pdf_path = Path(sys.argv[1]).expanduser().resolve()
    if not pdf_path.is_file():
        raise SystemExit(f"Input PDF not found: {pdf_path}")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required.")

    client = OpenAI(api_key=api_key)

    page_summaries: list[str] = []
    batch: list[tuple[int, bytes]] = []
    for index, image_bytes in enumerate(render_pdf_pages(pdf_path, DPI), start=1):
        batch.append((index, image_bytes))
        if len(batch) < CONCURRENCY:
            continue

        for page_number, summary in summarize_batch(client, batch):
            page_summaries.append(f"Page {page_number}\n{summary}")
            print(f"Summarized page {page_number}", flush=True)
        batch.clear()

    if batch:
        for page_number, summary in summarize_batch(client, batch):
            page_summaries.append(f"Page {page_number}\n{summary}")
            print(f"Summarized page {page_number}", flush=True)

    output_text = "\n\n".join(page_summaries).strip() + "\n"
    print(output_text, end="")


if __name__ == "__main__":
    main()
