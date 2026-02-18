# Mistral OCR

An experiment with the Mistral Document AI OCR API for extracting text and structured content from PDF documents.

## Overview

This project demonstrates how to use Mistral's `mistral-ocr-latest` model to:

- Upload PDF files to Mistral Cloud
- Extract text via OCR processing
- Retrieve structured content including tables (HTML format)
- Extract headers and footers
- Get embedded images as base64

## Setup

### Prerequisites

- Python 3.11+
- Mistral API key

### Installation

```bash
# Install with notebook support
uv sync --extra notebook

# Then register the kernel
uv run python -m ipykernel install --user --name=.venv
```

Open `cookbook.ipynb` in VS Code.

## Basic Usage

```python
from mistralai import Mistral
from pathlib import Path

client = Mistral(api_key="your_api_key")

# Upload PDF
path = Path("data/your_document.pdf")
uploaded_pdf = client.files.upload(
    file={"file_name": path.name, "content": open(path, "rb")},
    purpose="ocr"
)

# Get signed URL
signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

# Process OCR
ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={"type": "document_url", "document_url": signed_url.url},
    table_format="html",
    extract_header=True,
    extract_footer=True,
    include_image_base64=True
)

# Clean up
client.files.delete(file_id=uploaded_pdf.id)
```

## Resources

Check official Mistral AI [documentation](https://docs.mistral.ai/capabilities/document_ai/basic_ocr#ocr-images-and-pdfs) for more information.