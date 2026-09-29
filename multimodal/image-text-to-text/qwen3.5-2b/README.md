# Qwen3.5-2B

Spike: can [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), a small general video VLM, watch a security camera clip, describe what happens, and decide on its own whether an alert rule was triggered, staying quiet when nothing happens?

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- A Mac with Apple Silicon (the notebook runs the model on `mps`)

### Installation

```bash
uv sync
uv run python -m ipykernel install --user --name=qwen3.5-2b --display-name "Python (qwen3.5-2b)"
```

No system FFmpeg is needed: clips are decoded with OpenCV and passed to the processor as frames.

## Notebook

`qwen_security_spike.ipynb` runs one generic prompt (the camera's alert rules, a description, and a list of triggered rules) on ten short CCTV clips from [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/): five events (break-in, fight, shoplifting, vehicle theft, robbery), four normal scenes and one ambiguous arrest. It compares 2 fps, 2 fps with thinking, and 4 fps, and reports description quality, alerts, and time per clip.

The clips are downloaded into `data/` (git-ignored) from the [`mteb/ucf-crime`](https://huggingface.co/datasets/mteb/ucf-crime) mirror. UCF-Crime is released for research use.

## Resources

- [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/)
