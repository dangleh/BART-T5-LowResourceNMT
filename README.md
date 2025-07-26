# BART-T5 Low Resource NMT

A production-ready, demo-friendly Neural Machine Translation (NMT) system for English → Vietnamese using mBART50, fine-tuned for low-resource scenarios. The project provides training, inference, and a Streamlit web demo.

## Features

- Fine-tuning and inference with HuggingFace Transformers (mBART50)
- Streamlit web app for interactive translation demo
- Docker & docker-compose support for easy deployment
- Modular, extensible codebase

## Project Structure

```text
├── app.py                # Streamlit web app
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image for app
├── docker-compose.yml    # Compose for resource limits, volume, port
├── scripts/              # Shell scripts for training/inference
├── src/                  # Source code
│   ├── data.py           # Data loading & preprocessing
│   ├── evaluate.py       # Evaluation metrics
│   ├── infer.py          # Inference utilities
│   ├── model.py          # Model & tokenizer loading
│   ├── train.py          # Training pipeline
│   └── ...
└── outputs/              # Model checkpoints, outputs
```

## Quickstart

### 1. Build & Run with Docker

```bash
docker-compose up --build
```

App will be available at: [http://localhost:8501](http://localhost:8501)

### 2. Local Run (Python 3.10+)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Training

Edit `src/train.py` and run:

```bash
python src/train.py
```

## Inference

Use the Streamlit app or run:

```bash
python src/infer.py
```

## Model

- Default: `dangleh/mbart50_envi_finetuned` (can be changed in `src/model.py`)
- Model artifacts stored in `outputs/mbart50_envi_finetuned/model/`

## Customization

- Update data loading in `src/data.py`
- Adjust training parameters in `src/train.py`

## Requirements

- Python 3.10+
- Docker (optional)
