# Image Classification with Transfer Learning

Fine-tune pre-trained convolutional neural networks for custom image classification tasks.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Download Images](./[1]%20Download%20Images.ipynb) | Fetch and organize training images |
| 2 | [Data Pre-Processing](./[2]%20Data%20Pre-Processing.ipynb) | Augmentation, normalization, dataset splits |
| 3 | [Train Model](./[3]%20Train%20Model.ipynb) | Fine-tune pre-trained CNN backbone |
| 4 | [Upload Model](./[4]%20Upload%20Model.ipynb) | Save and upload model to Hugging Face Hub |
| 5 | [Make Inference](./[5]%20Make%20Inference.ipynb) | Load model and run predictions |

## Approach

1. Start with a pre-trained model (e.g., ResNet, EfficientNet)
2. Replace the classification head for the target classes
3. Train with frozen backbone, then fine-tune end-to-end
4. Export for inference

## Usage

Run notebooks sequentially (1-5) for a complete pipeline from data collection to deployment.
