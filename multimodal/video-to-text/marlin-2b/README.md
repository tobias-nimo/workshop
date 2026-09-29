# Marlin-2B

Feasibility spike for running [Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B), a 2B video VLM with `caption` and `find` modes, on Apple Silicon.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Access to the gated model: request it on the [model page](https://huggingface.co/NemoStation/Marlin-2B) (auto-approved)

### Installation

```bash
uv sync
uv run hf auth login
uv run python -m ipykernel install --user --name=marlin-2b --display-name "Python (marlin-2b)"
```

No system FFmpeg is needed: the notebook decodes clips with OpenCV and passes frames to the processor directly.

## Notebook

`marlin_spike.ipynb` loads Marlin on MPS, applies Marlin's training-time preprocessing (2 fps, ~200k pixels per frame), and tests:

- full-frame vs. cropped captions on a 4K clip,
- whether `find` can tell events that happen from events that don't (it can't: it always returns a span),
- caption and find latency.

Sample videos are downloaded to `data/` (git-ignored).

## Resources

- [Marlin-2B model card](https://huggingface.co/NemoStation/Marlin-2B)
