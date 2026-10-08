# Lung Cancer Survey Classification — Final Report

An educational machine-learning application that compares decision-tree, KNN and Random Forest classifiers on a lung-cancer survey dataset.

> This is an academic classification exercise, not a diagnostic or clinical risk-assessment tool. Model outputs are not medical advice or validated disease probabilities.

## Relationship to the other repository

At review time, `app.py` and `train_models.py` are identical to those in [LungCancerApp](https://github.com/bigbaboy/LungCancerApp). These are two repository versions of the same academic topic, not two independent projects.

## What is implemented

- A Vietnamese Streamlit form for survey inputs.
- A saved preprocessing pipeline loaded from `models/pipeline.pkl`.
- Selection among an entropy-based decision tree, KNN and Random Forest.
- Predicted class output and model probability output where available.
- A training script that prepares data, performs a stratified train/test split and prints accuracy.

The UI calls the decision-tree option “ID3.” The code actually uses scikit-learn's `DecisionTreeClassifier(criterion='entropy')`; it is not a standalone implementation of the original ID3 algorithm.

## Files

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface and inference |
| `train_models.py` | Preprocessing, training and artifact export |
| `models/pipeline.pkl` | Serialized input transformation |
| `models/id3_model.pkl` | Decision-tree artifact |
| `models/knn_model.pkl` | KNN artifact |
| `models/rf_model.pkl` | Random Forest artifact |

## Local inference demo

```bash
git clone https://github.com/bigbaboy/Final-report.git
cd Final-report
python -m venv .venv
```

Activate the environment, then:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The `requirements.txt` file in this snapshot is empty. Install compatible versions of Streamlit, pandas, NumPy and scikit-learn before running; the original training environment has not been recorded.

The repository includes saved model artifacts. Pickle loading requires trusted files and compatible library versions; the unpinned requirements do not establish an exact training environment. The artifacts were not executed during this documentation update.

## Retraining prerequisites

The training script currently reads `D://Download//survey lung cancer.csv`. The dataset is not included in the reviewed repository. To retrain, obtain the original dataset with appropriate permission and update that path, preserving the expected input columns.

The training pipeline uses gender encoding, age groups, one-hot encoding and MinMax scaling. It fits an entropy-based decision tree, a seven-neighbor classifier and a 100-tree Random Forest. Accuracy is printed by the script, but no reproducible evaluation output is bundled in this documentation.

## Known technical limits

- KNN uses a hard-coded slice starting at transformed column 8; feature names and ordering should be verified.
- Age-bin boundaries and unknown inputs require explicit tests.
- Dataset provenance, dependency versions and evaluation artifacts need fuller documentation.
- External validation, calibration and clinical evaluation have not been established.

## Next steps

Make the dataset path configurable, record transformed feature names, lock compatible dependencies and save a reproducible evaluation report. Document project contributors and responsibilities separately from the functionality described above.
