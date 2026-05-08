"""
Mistral OCR CLI.

Convert any supported document or image into Markdown using the Mistral
Document AI OCR API. Extracted images are decoded and saved alongside
the Markdown file.

Output layout (default, next to the input file):

    # When the document contains figures:
    <input_dir>/<stem>/
        <stem>.md
        figures/
            img-0.jpeg
            ...

    # When the document contains no figures:
    <input_dir>/<stem>.md

API limits: max 50 MB per file, max 1,000 pages.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Annotated

import typer
from mistralai import Mistral


IMAGE_MIME_TYPES: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".avif": "image/avif",
    ".tiff": "image/tiff",
    ".tif":  "image/tiff",
    ".gif":  "image/gif",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".bmp":  "image/bmp",
    ".webp": "image/webp",
}

DOCUMENT_MIME_TYPES: dict[str, str] = {
    ".pdf":   "application/pdf",
    ".docx":  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc":   "application/msword",
    ".pptx":  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt":   "application/vnd.ms-powerpoint",
    ".xlsx":  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv":   "text/csv",
    ".txt":   "text/plain",
    ".epub":  "application/epub+zip",
    ".xml":   "application/xml",
    ".rtf":   "application/rtf",
    ".odt":   "application/vnd.oasis.opendocument.text",
    ".bib":   "text/plain",
    ".fb2":   "application/xml",
    ".ipynb": "application/json",
    ".tex":   "text/x-tex",
    ".opml":  "text/x-opml",
    ".1":     "text/troff",
    ".man":   "text/troff",
}


def detect_document_type(path: Path) -> str:
    ext = path.suffix.lower()

    if ext in IMAGE_MIME_TYPES:
        return "image_url"
    if ext in DOCUMENT_MIME_TYPES:
        return "document_url"

    raise typer.BadParameter(
        f"Unsupported file extension '{ext}'. "
        f"Supported images: {', '.join(sorted(IMAGE_MIME_TYPES))}. "
        f"Supported documents: {', '.join(sorted(DOCUMENT_MIME_TYPES))}."
    )


def build_document_payload(path: Path, doc_type: str) -> dict:
    ext = path.suffix.lower()
    b64 = base64.standard_b64encode(path.read_bytes()).decode()

    if doc_type == "image_url":
        media_type = IMAGE_MIME_TYPES[ext]
        return {
            "type": "image_url",
            "image_url": f"data:{media_type};base64,{b64}",
        }

    media_type = DOCUMENT_MIME_TYPES[ext]
    return {
        "type": "document_url",
        "document_url": f"data:{media_type};base64,{b64}",
    }


def save_images(pages, output_dir: Path) -> int:
    figures_dir = output_dir / "figures"
    saved = 0
    for page in pages:
        for img in page.images:
            if not img.image_base64:
                continue

            raw_b64 = img.image_base64
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]

            if saved == 0:
                figures_dir.mkdir(exist_ok=True)
            (figures_dir / img.id).write_bytes(base64.b64decode(raw_b64))
            saved += 1
    return saved


def build_markdown(pages) -> str:
    parts = []
    for page in pages:
        md = (page.markdown or "").strip()
        if not md:
            continue
        for img in page.images:
            md = md.replace(f"]({img.id})", f"](figures/{img.id})")
        parts.append(md)
    return "\n\n---\n\n".join(parts)


app = typer.Typer(
    add_completion=False,
    help="Convert documents and images to Markdown via the Mistral OCR API.",
)


@app.command()
def main(
    file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to the document or image to convert.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o",
            help=(
                "Output directory. By default, writes <input_dir>/<stem>.md when the "
                "document has no figures, or <input_dir>/<stem>/ when it does."
            ),
        ),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress progress output."),
    ] = False,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key", "-k",
            envvar="MISTRAL_API_KEY",
            show_envvar=True,
            help="Mistral API key. Falls back to the MISTRAL_API_KEY environment variable.",
        ),
    ] = None,
) -> None:
    """Convert FILE to Markdown using Mistral OCR."""
    if not api_key:
        typer.secho(
            "Error: provide --api-key or set the MISTRAL_API_KEY environment variable.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    explicit_output = output.resolve() if output is not None else None

    doc_type = detect_document_type(file)
    document = build_document_payload(file, doc_type)

    if not quiet:
        typer.echo(f"Processing {file.name} ({doc_type})...")

    client = Mistral(api_key=api_key)
    try:
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document=document,
            include_image_base64=True,
        )
    except Exception as e:
        typer.secho(f"OCR request failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    has_figures = any(
        img.image_base64 for page in response.pages for img in page.images
    )

    if explicit_output is not None:
        target_dir = explicit_output
    elif has_figures:
        target_dir = file.parent / file.stem
    else:
        target_dir = file.parent

    target_dir.mkdir(parents=True, exist_ok=True)
    md_path = target_dir / f"{file.stem}.md"

    image_count = save_images(response.pages, target_dir) if has_figures else 0
    md_path.write_text(build_markdown(response.pages), encoding="utf-8")

    if not quiet:
        typer.echo(
            f"Wrote {md_path} ({len(response.pages)} pages, {image_count} images)."
        )
    typer.echo(str(md_path))


if __name__ == "__main__":
    app()
