# Automated Defect Detection System for Quality Assurance in Manufacturing

## Background

This project started as a classroom assignment for IENG 493C where our team built a basic image classifier to detect defects in cell phone cases coming off a conveyor belt. The original version used TensorFlow/Keras with a simple two-layer CNN trained on 38 images and achieved 100% accuracy on the augmented test set. While that was a good starting point, I knew there were significant gaps between what we built in class and what a real manufacturing facility would actually need. This repository is my attempt to bridge that gap.

The original classroom project is preserved in the `Classroom Project/` folder for reference.

## The Business Problem

The CP-Lab manufacturing facility had no automated quality inspection system. Operators were visually checking cell phone main cases for surface defects like scratches and excess material. This manual process is slow, inconsistent, and does not scale. A single missed defect that reaches a customer costs significantly more than a false alarm that pulls a good part off the line for re-inspection. That asymmetry between the cost of a false negative versus a false positive became a central design constraint for this project.

## What I Built

I rebuilt the entire pipeline from scratch using PyTorch and structured it as a deployable application rather than a notebook experiment.

### Architecture

```
data/                    Raw images (16 good, 21 bad parts)
src/
  config.py              Centralized hyperparameters and cost settings
  dataset.py             PyTorch Dataset with augmentation and weighted sampling
  model.py               ResNet18 fine-tuned classifier + convolutional autoencoder
  train.py               Cost-sensitive training loop with early stopping
  evaluate.py            Evaluation plots (confusion matrix, PR curve)
  autoencoder.py         Anomaly detection trained only on good parts
  api.py                 FastAPI inference endpoint
  dashboard.py           Streamlit web dashboard
run_pipeline.py          End-to-end training script
Dockerfile               Container image definition
docker-compose.yml       Multi-service orchestration (API + Dashboard)
```

### Key Technical Decisions

**Transfer Learning with ResNet18** -- With only 37 images, training a CNN from scratch is not realistic. I used a ResNet18 backbone pretrained on ImageNet, froze the early layers, and fine-tuned `layer4` along with a new fully connected head. This lets the model leverage general visual features while adapting the final representations to our specific defect patterns.

**Aggressive Data Augmentation** -- Random flips, rotations (up to 30 degrees), color jitter, affine transforms, and Gaussian blur are applied on every training pass. Combined with 4x oversampling through a `WeightedRandomSampler`, this gives the model a much more diverse view of the data than the raw 37 images would suggest.

**Cost-Sensitive Loss** -- I weighted false negatives at 5x the cost of false positives in the training loss. In manufacturing QA, letting a defective part through is far more expensive than flagging a good part for re-inspection. This is reflected in the cost-sensitive confusion matrix that the evaluation generates.

**Convolutional Autoencoder for Novel Defects** -- The supervised classifier can only catch defect types it has seen. To handle novel or unseen defect types, I trained a convolutional autoencoder exclusively on good parts. At inference time, if a part has high reconstruction error (above the 95th percentile threshold from training), it gets flagged as a potential novel defect. This is important in manufacturing where new failure modes can emerge without warning.

**Deployment as Microservices** -- The model is wrapped in a FastAPI backend that accepts image uploads and returns a JSON response with the classification label, defect probability, reconstruction error, and whether a novel defect is suspected. A Streamlit dashboard provides a simple UI where an operator can upload a photo and get an instant PASS/FAIL verdict. Both services are containerized with Docker Compose.

## What I Learned

**Small datasets are hard.** The classroom version reported 100% accuracy, but that was on a tiny test split with heavy augmentation applied uniformly. When I restructured the splits properly and evaluated more carefully, the real performance was closer to 75%. This taught me that metrics on small datasets should be interpreted cautiously and that reporting precision-recall curves gives a much more honest picture than a single accuracy number.

**Transfer learning is essential for small data regimes.** Freezing the entire backbone and only training the FC layer gave poor results (62% accuracy). Unfreezing the last residual block (`layer4`) made a significant difference because the model could adapt higher-level feature representations to the domain while still leveraging low-level features learned from ImageNet.

**Cost asymmetry matters.** In a real manufacturing setting, not all errors are equal. A standard binary cross-entropy loss treats false positives and false negatives identically. Incorporating cost-sensitive weights into the loss function and visualizing the cost-weighted confusion matrix made me think about the problem from the business perspective rather than just optimizing a metric.

**Anomaly detection complements classification.** A supervised classifier is limited to the defect types present in the training set. The autoencoder approach gives a safety net for catching things the classifier has never seen. This is a pattern I had read about but implementing it myself made the tradeoffs concrete -- the threshold selection is subjective and the reconstruction error distribution depends heavily on the autoencoder architecture.

**Going from notebook to application is a real engineering effort.** The classroom notebook was about 200 lines in a single file. The production version has separate modules for data loading, modeling, training, evaluation, serving, and deployment. Writing the Dockerfile, handling service-to-service communication in Docker Compose, and making the API robust to different image formats were all things that do not show up in a Jupyter notebook but matter for deployment.

## Results

| Metric | Classroom (TF/Keras) | This Project (PyTorch) |
|--------|---------------------|----------------------|
| Framework | TensorFlow 2.x | PyTorch 2.x |
| Model | 2-layer CNN from scratch | ResNet18 (fine-tuned) |
| Training Images | 38 (augmented to 114) | 37 (4x oversampled with augmentation) |
| Loss Function | Binary Cross-Entropy | Cost-Sensitive BCE (FN=5x) |
| Novel Defect Detection | None | Convolutional Autoencoder |
| Deployment | Jupyter Notebook | FastAPI + Streamlit + Docker |
| Evaluation | Accuracy only | Precision-Recall Curve + Cost Matrix |

## How to Run

### Training

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

This trains the classifier, generates evaluation plots in `outputs/plots/`, and trains the autoencoder. Model weights are saved to `outputs/models/`.

### Docker Deployment

```bash
docker compose up --build
```

- API: http://localhost:8000 (Swagger docs at `/docs`)
- Dashboard: http://localhost:8501

### API Usage

```bash
curl -X POST http://localhost:8000/predict -F "file=@path/to/image.jpg"
```

Returns:
```json
{
  "label": "FAIL",
  "defect_probability": 0.7974,
  "reconstruction_error": 0.001978,
  "anomaly_threshold": 0.001974,
  "novel_defect_suspected": true
}
```

## Business Impact

For a facility like CP-Lab, this system addresses three concrete problems:

1. **Consistency** -- Automated inspection eliminates operator fatigue and subjective judgment. Every part is evaluated against the same threshold.
2. **Cost Reduction** -- By weighting the model to be aggressive about catching defects (at the cost of some false alarms), the system minimizes the expensive scenario of shipping defective parts to customers.
3. **Adaptability** -- The autoencoder layer means the system can flag new defect types that were not in the original training data, giving quality engineers an early warning when the manufacturing process drifts.

The system is designed to be non-invasive -- it uses an overhead camera on the existing conveyor belt and does not require any changes to the physical production line.

## Future Improvements

- Collect a larger and more diverse dataset to improve generalization
- Experiment with other pretrained backbones (EfficientNet, MobileNet) for faster inference
- Add a feedback loop where flagged parts are reviewed and used to retrain the model periodically
- Integrate directly with the conveyor PLC for automated part rejection

## Contributors

- Akshay Patel

Original classroom project contributors: Akshay Patel, Sagar Pranthi, Astik Sharma

## Acknowledgments

Thanks to Mackenzie Keepers and the IENG 493C course for the original project that inspired this work.
