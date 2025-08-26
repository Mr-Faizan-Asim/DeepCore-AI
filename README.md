# Deep-AI

> A collection of machine learning & deep learning projects and notebooks — regression, classification, explainable AI, and NLP fine‑tuning experiments.

**Author:** Muhammad Faizan Asim ([mr.faizan.asim@gmail.com](mailto:mr.faizan.asim@gmail.com))
**Kaggle:** [Muhammad Faizan Asim](https://www.kaggle.com/muhammadfaizanasim)

---

## Overview

This repository (Deep-AI) contains a set of data science and deep learning projects collected as reproducible Jupyter notebooks and small helper scripts. Projects range from classical regression and PCA analyses to explainable AI (XAI) notebooks and a Hugging Face BERT fine‑tuning workflow for text classification.

Use this README to quickly understand each component, how to run the notebooks locally/Colab/Kaggle, and how to reproduce training and evaluation steps.

---

## Repo structure (high level)

```
Deep-AI/
├─ Linear Regression Project of Price Predication/   # linear regression project(s)
├─ MultiVariablePredication/                         # multivariable regression
├─ DiaXAi_MODEL_+_XAI.ipynb                           # model + explainability notebook
├─ Email.py                                          # small utility / email helper
├─ Principal component analysis (PCA).py              # PCA script
├─ data-preprocessing-feature-engineering.ipynb      # preprocessing & feature engineering
├─ dna-change-prediction-using-decisiontree*.ipynb    # DNA change / decision tree experiments
├─ dna-classification-and-disease-risk-prediction.ipynb
├─ fine-tuning-bert-with-hugging-face-for-text-cl...ipynb  # BERT fine-tune notebook (text classification)
├─ tuberculosis-classification-from-chest-x-ray.ipynb
└─ README.md
```

> Filenames in the repo are descriptive — open the notebooks to see dataset links, hyperparameters and results.

---

## Quickstart — recommended workflow

1. Clone this repository:

```bash
git clone https://github.com/Mr-Faizan-Asim/Deep-AI.git
cd Deep-AI
```

2. Create an isolated Python environment (recommended):

```bash
# using venv
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
.\.venv\Scripts\activate   # Windows (PowerShell: .\.venv\Scripts\Activate.ps1)

# or using conda
conda create -n deepai python=3.10 -y
conda activate deepai
```

3. Install dependencies. Create a `requirements.txt` file (if not present) with key packages below and then:

```bash
pip install -r requirements.txt
```

Suggested minimum `requirements.txt`:

```
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyterlab
notebook
xgboost
lightgbm
tensorflow
torch
torchvision
transformers
datasets
accelerate
evaluate
opencv-python
lime
shap
scikit-image
```

> Note: GPU-accelerated packages (TensorFlow/PyTorch with CUDA) may require additional platform-specific installation.

---

## Running notebooks (choices)

### Option A — Run locally

1. Activate the environment.
2. Start JupyterLab:

```bash
jupyter lab
```

3. Open and run the desired notebook (e.g. `fine-tuning-bert-with-hugging-face-for-text-cl...ipynb`).

### Option B — Run on Google Colab

1. Upload the notebook to Colab (Open → Upload notebook) or open directly from GitHub using Colab’s GitHub integration.
2. If using large datasets, prefer mounting Google Drive or loading from Kaggle.

### Option C — Run on Kaggle Notebooks

Many notebooks were authored for Kaggle — you can open them directly on Kaggle and run with Kaggle datasets and free GPU/TPU.

---

## Notebook / file descriptions

A short description for each key file:

* `Linear Regression Project of Price Predication/` — Project folder for housing a classical price prediction pipeline using linear regression and model evaluation.
* `MultiVariablePredication/` — Multivariate regression experiments using multiple features and feature selection.
* `data-preprocessing-feature-engineering.ipynb` — Utilities and patterns to clean raw datasets, impute missing values, encode categorical variables and create features.
* `DiaXAi_MODEL_+_XAI.ipynb` — Model training combined with explainability (SHAP, LIME) visualizations demonstrating how predictions are influenced by features.
* `Principal component analysis (PCA).py` — Script showing dimensionality reduction examples and how to visualize principal components for high-dimensional data.
* `dna-change-prediction-using-decisiontree...ipynb` and `dna-classification-and-disease-risk-prediction.ipynb` — Bioinformatics classification experiments (DNA change prediction, disease risk classifiers) — include data preprocessing, model training and evaluation.
* `fine-tuning-bert-with-hugging-face-for-text-cl...ipynb` — Hugging Face Transformers fine-tuning notebook (BERT) for text classification. Includes tokenization, dataset preparation, Trainer/TrainingArguments or `accelerate` setup, and evaluation metrics.
* `tuberculosis-classification-from-chest-x-ray.ipynb` — CNN-based chest X‑ray image classification for TB detection (data loading, augmentation, training loop and metrics).
* `Email.py` — Small helper: likely used to send automated emails (double-check credentials and never commit secrets). If it uses SMTP credentials, move them to environment variables.

---

## Reproducibility tips and large files

* **Datasets & large model files:** Do not commit large datasets or model checkpoints directly to Git. Use one of:

  * Provide direct download links in each notebook (Kaggle dataset links, Google Drive public links).
  * Use Git LFS for large binary files.
  * Document where to place files locally (e.g. `data/` folder) and how to name them.

* **Secrets & credentials:** Never push API keys, tokens or email credentials. Use environment variables or a `.env` file added to `.gitignore`.

---

## BERT fine-tuning (short actionable guide)

If you want to reproduce the BERT fine-tuning notebook quickly:

1. Ensure these packages are installed: `transformers`, `datasets`, `accelerate`, `evaluate`.
2. Prepare a CSV/JSON dataset with `text` and `label` columns.
3. Minimal example commands inside a notebook:

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)

dataset = load_dataset('csv', data_files={'train':'train.csv','validation':'val.csv'})

def preprocess(ex):
    return tokenizer(ex['text'], truncation=True, padding='max_length')
dataset = dataset.map(preprocess, batched=True)

args = TrainingArguments(
    output_dir='./outputs',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
)

trainer.train()
```

> For larger scale runs use `accelerate` and a GPU runtime.

---

## Good to know / best practices

* Add a `requirements.txt` and an optional `environment.yml` (conda) to help others reproduce your environment.
* Add a `LICENSE` file (MIT is a common permissive choice).
* Add a `CONTRIBUTING.md` if you want external contributors.
* Add small sample datasets (or scripts to download them) so users can run notebooks quickly.

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes and open a Pull Request

Please include reproducible steps and small datasets or pointers to data when submitting PRs that touch notebooks.

---

## License

This repository does not include an explicit license file yet. If you are happy to make it open source, consider adding `LICENSE` with the MIT license.

---

## Contact

Muhammad Faizan Asim — [mr.faizan.asim@gmail.com](mailto:mr.faizan.asim@gmail.com)
GitHub: [Mr-Faizan-Asim](https://github.com/Mr-Faizan-Asim)
Kaggle: [Muhammad Faizan Asim](https://www.kaggle.com/muhammadfaizanasim)


