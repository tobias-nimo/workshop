# Text Classification with BERT

Fine-tune BERT models for single-label and multi-label text classification tasks.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [single_label_bert.ipynb](./single_label_bert.ipynb) | Standard classification (one class per sample) |
| [multi_label_bert.ipynb](./multi_label_bert.ipynb) | Multi-label classification (multiple classes per sample) |

## Datasets

Located in `data/`:
- `assistant_commands.csv` - Command intent classification
- `research_papers.csv` - Paper topic classification

## Key Differences

### Single-Label
- Softmax activation
- Cross-entropy loss
- One predicted class per input

### Multi-Label
- Sigmoid activation
- Binary cross-entropy loss
- Multiple predicted classes per input (threshold-based)

## Requirements

```
transformers
torch
datasets
scikit-learn
pandas
```
