# Hateful-memes-classification

# Hateful Memes Classification

This project focuses on identifying hateful content in memes by combining computer vision and natural language processing techniques. With the rise in social media usage, memes have become a powerful tool for both humor and harm. This work aims to detect memes that potentially spread hate by leveraging deep learning and explainable AI (XAI) methods.

## Project Summary

We use multimodal learning to fuse visual and textual information for meme classification. Our pipeline includes:
- Dataset parsing and cleaning (Facebook Hateful Memes dataset in JSONL format).
- Visualizations for understanding class distributions, text length, and token frequency.
- Multiple deep learning models trained and compared.
- Explainability modules for transparency and interpretability.

## Dataset

The dataset consists of memes scraped from Facebook and annotated with binary labels:
- `0`: Non-hateful
- `1`: Hateful

Each instance includes:
- An image (`.jpg`)
- Overlaid text
- A label

Text is tokenized, and images are preprocessed before being passed to the models.

## Models Trained

### 1. ResNet50 + BERT
- Image features extracted using pretrained ResNet50 (2048-dim).
- Text encoded using BERT (768-dim).
- Features concatenated and passed through a two-layer MLP classifier.
- Achieved **60.5% accuracy**, **0.605 ROC-AUC**.

### 2. EfficientNet-B0 + DistilBERT
- More lightweight model with EfficientNet-B0 (1280-dim) and DistilBERT.
- Improved both speed and performance.
- Achieved **62.35% accuracy**, **0.625 ROC-AUC**.

### 3. CLIP (ViT-B/32)
- Leveraged OpenAI's CLIP for zero-shot classification using cosine similarity in a shared embedding space.
- Best performing model.
- Achieved **69.63% accuracy**, **0.706 ROC-AUC**.

## Results Summary

| Model                     | Accuracy | ROC-AUC | F1 Score (macro) |
|---------------------------|----------|---------|------------------|
| ResNet50 + BERT           | 60.5%    | 0.605   | 0.56             |
| EfficientNet + DistilBERT | 62.4%    | 0.625   | 0.58             |
| CLIP                      | 69.6%    | 0.707   | 0.61             |

- **CLIP** was the most effective out-of-the-box model.
- **EfficientNet + DistilBERT** offered a good trade-off between speed and accuracy.
- **ResNet50 + BERT** performed the lowest but established a solid baseline.

## Explainable AI (XAI)

To gain insights into what influenced the model predictions, we applied:

### 1. Integrated Gradients
- Captures attribution by integrating gradients along a path from a baseline to input.
- Highlights influential words like "no" or "run".

### 2. Grad-CAM
- Produces heatmaps over image regions that influence predictions.
- Useful for visualizing which parts of the image (e.g., knee or facial features) contributed to decisions.

### 3. Saliency Maps
- Calculates gradients of output w.r.t. inputs.
- Highlights sensitivity in pixel space.

Each model had inferences documented showing token-level attributions and visual heatmaps.

## How to Run

Clone this repository and install the requirements:
```bash
git clone https://github.com/yourusername/hateful-memes-classification.git
cd hateful-memes-classification
pip install -r requirements.txt
