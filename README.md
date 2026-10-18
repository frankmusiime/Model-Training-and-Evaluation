# Adult Census Income Classification

This project uses the UCI Adult Census Income dataset to predict whether a person's income is:

- `<=50K`
- `>50K`

The notebook compares traditional machine-learning models with a TensorFlow neural network.

## Project files

- `Income334_classifier.ipynb` - concise authoritative submission notebook
- `adult.data` - official training data
- `adult.test` - official test data
- `requirements.txt` - Python package requirements for submission
- `environment.txt` - equivalent local environment specification
- `artifacts/` - saved models, preprocessing, and result tables
- `REPORT_DRAFT.md` - report draft based on the executed notebook

## Environment setup

The project uses Python 3.13 and a local virtual environment.

From this folder, run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name adult-income --display-name "Adult Income (Python 3.13)"
```

Open the notebook in VS Code or Jupyter and select the kernel:

```text
Adult Income (Python 3.13)
```

The notebook uses the local dataset files. It does not require a Google Colab or Google Drive mount.

For TensorBoard, run:

```powershell
.\.venv\Scripts\tensorboard.exe --logdir logs\adult_income_nn
```

## Workflow

The submission notebook contains only the required assessment workflow. Exploratory history is preserved separately in `Income334_classifier_full_history.ipynb` and is not needed to reproduce the reported results.

The notebook:

1. Loads the original training and test files.
2. Removes the trailing period from labels in `adult.test`.
3. Changes `?` values to missing values.
4. Imputes missing categorical values as `Unknown`.
5. Imputes numerical missing values using a median learned from the fitting data.
6. One-hot encodes categorical features and standardizes numerical features.
7. Splits the official training data into an 80% fitting set and a 20% validation set.
8. Fits the preprocessor only on the fitting set.
9. Compares traditional models and a neural network on the same validation data.
10. Selects the final model using validation performance.
11. Evaluates the selected model once on the official test set.

This prevents information from the validation or test data being used to fit preprocessing.

## Experiments

The authoritative section contains eight distinct experiments:

- Logistic Regression with balanced class weights and `C=0.1`
- Logistic Regression with balanced class weights and `C=1.0`
- Random Forest with balanced class weights and 200 trees
- Random Forest with balanced class weights and `min_samples_leaf=2`
- Gradient Boosting with 100 trees and learning rate `0.10`
- Gradient Boosting with 200 trees and learning rate `0.05`
- Gradient Boosting with 300 trees, learning rate `0.05`, and depth `2`
- Neural network with hidden layers of 64 and 32 units plus dropout

The main selection metric is F1-score for the `>50K` class. Accuracy, precision, recall, and ROC-AUC are also reported.

## Verified results

The best validation experiment was:

```text
Random Forest | balanced | 200 trees
```

Validation results:

| Metric | Value |
|---|---:|
| Accuracy | 0.8465 |
| Precision for `>50K` | 0.6562 |
| Recall for `>50K` | 0.7608 |
| F1-score for `>50K` | 0.7047 |
| ROC-AUC | 0.9082 |

After selecting the model on validation data, the final official-test results were:

| Metric | Value |
|---|---:|
| Accuracy | 0.8404 |
| Precision for `>50K` | 0.6419 |
| Recall for `>50K` | 0.7332 |
| F1-score for `>50K` | 0.6845 |
| ROC-AUC | 0.9016 |

These values are calculated from model predictions and are saved in `artifacts/`.

## Data-quality decisions

The training file contains 24 exact duplicate rows. They were retained because the supplied Adult dataset is evaluated as provided. The notebook reports the duplicates clearly instead of silently removing them.

Missing-value markers are preserved as a meaningful data issue. Categorical missing values are represented as `Unknown`, while numerical values use a fitting-set median.

## Saved artifacts

The notebook saves:

- `adult_income_preprocessor_fit_split.joblib` - preprocessing fitted on the fitting data
- `adult_income_selected_traditional_pipeline.joblib` - selected model with its preprocessing
- `adult_income_neural_network.keras` - saved neural-network model
- `validation_results.csv` - validation comparison table
- `experiment_metadata.csv` - experiment settings, splits, and traceability information
- `final_official_test_results.csv` - final test metrics

The notebook also reloads the saved artifacts and performs a smoke test.

## TensorBoard

The authoritative neural-network run writes logs to:

```text
logs/adult_income_nn/authoritative_64_32_dropout
```

To view the logs from this folder, run:

```powershell
.\.venv\Scripts\tensorboard.exe --logdir logs\adult_income_nn
```

## Reproducibility

The notebook uses random seed `42` for NumPy, TensorFlow, model configuration, and data splitting where supported.

## Submission checklist

The notebook, requirements files, generated result tables, saved model artifacts, report-ready figures, and report draft are present.

Before submission, also confirm the course-specific requirements for:

- Google Drive copies or links
- Written report
- Demonstration video
- Any required screenshots or presentation files
