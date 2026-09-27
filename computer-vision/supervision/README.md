# Supervision

Object detection and tracking examples on videos using [Roboflow Supervision](https://supervision.roboflow.com/latest) with [RF-DETR](https://github.com/roboflow/rf-detr) models.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
uv sync
uv run python -m ipykernel install --user --name=supervision --display-name "Python (supervision)"
```

Then open the notebooks in VS Code (select the `.venv` kernel) or launch Jupyter Lab:

```bash
uv run jupyter lab
```

## Notebooks

| Notebook | Description |
| --- | --- |
| `people_walking_tracking.ipynb` | Detect people with RF-DETR, use tiled inference (`sv.InferenceSlicer`) to catch small people, and track them with ByteTrack from the `trackers` package. |

Sample videos are downloaded to `data/` and annotated videos are written to `output/` (both git-ignored).

## Resources

- [Supervision documentation](https://supervision.roboflow.com/latest)
- [RF-DETR documentation](https://rfdetr.roboflow.com)
