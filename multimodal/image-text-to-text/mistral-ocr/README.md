# Mistral OCR

A small CLI that converts documents and images to Markdown using the Mistral Document AI OCR API (`mistral-ocr-latest`). Extracted images are decoded from base64 and saved alongside the Markdown output.

## Setup

### Prerequisites

- Python 3.11+
- A Mistral API key, exported as `MISTRAL_API_KEY`

### Installation

```bash
uv sync
export MISTRAL_API_KEY="your_api_key"   # or pass --api-key on the command line
```

This installs the `mistral-ocr` console script.

## Usage

```bash
# Convert a PDF (or DOCX, PPTX, image, etc.) to Markdown
uv run mistral-ocr data/LoRA.pdf

# Choose a custom output directory
uv run mistral-ocr data/LoRA.pdf --output ./out

# Suppress progress output (only the output path is printed on stdout)
uv run mistral-ocr data/LoRA.pdf --quiet

# Pass the API key explicitly instead of using the environment variable
uv run mistral-ocr data/LoRA.pdf --api-key "your_api_key"
```

By default the output goes next to the input file. The layout depends on whether the document contains any figures:

```
# When the document has figures, a folder is created:
data/
  LoRA.pdf
  LoRA/
    LoRA.md
    figures/
      img-0.jpeg
      ...

# When the document has no figures, the .md is written directly:
data/
  notes.txt
  notes.md
```

If `--output DIR` is given, everything is written into `DIR/` (the `figures/` subfolder is only created when needed).

The Markdown file concatenates every page (separated by `---`) and rewrites image references to point at `figures/`.

### Supported formats

- **Documents:** PDF, DOCX, DOC, PPTX, PPT, XLSX, CSV, TXT, EPUB, XML, RTF, ODT, BIB, FB2, IPYNB, TEX, OPML, man/troff pages.
- **Images:** JPEG, PNG, AVIF, TIFF, GIF, HEIC/HEIF, BMP, WebP.

### API limits

Files must not exceed 50 MB or 1,000 pages.

## Notebook demo

`cookbook.ipynb` contains a minimal end-to-end demonstration of the underlying Mistral OCR API (upload → signed URL → OCR call) for users who want to explore the raw response. To run it:

```bash
uv sync --extra notebook
uv run python -m ipykernel install --user --name=.venv
```

Then open `cookbook.ipynb` in VS Code or Jupyter.

## Resources

- [Mistral Document AI documentation](https://docs.mistral.ai/capabilities/document_ai/basic_ocr#ocr-images-and-pdfs)
