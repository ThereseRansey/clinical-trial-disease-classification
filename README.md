# Clinical Trial Disease Category Classification Using NLP and Machine Learning

An end-to-end NLP and machine learning project that reads the **Brief Summary** of a clinical trial and predicts which of **8 disease categories** it belongs to. It includes data cleaning, exploratory analysis, TF-IDF feature engineering, comparison of four classification algorithms, a prediction system with confidence scores, and an interactive Streamlit web app.

**Domain:** Healthcare Analytics, Medical NLP and Clinical Trial Intelligence Systems


## Results

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Linear SVM | 95.5% | 95.4% | 94.7% | 95.0% |
| **Logistic Regression** (final model) | **95.2%** | 94.8% | 94.4% | **94.6%** |
| Random Forest | 94.0% | 95.2% | 92.2% | 93.5% |
| Naive Bayes | 88.2% | 92.3% | 77.2% | 81.5% |

- The deployed model is **Logistic Regression**. Linear SVM scored slightly higher (under 0.5 points), but Logistic Regression was chosen because it gives probabilities directly (used as the confidence score) and is simple to explain. It reached **95.2% accuracy** and **0.946 macro F1** on 12,013 unseen test trials.
- **5-fold cross-validation:** about 95.2% accuracy (plus or minus 0.08%), so the result is stable.
- **Confidence works:** predictions with at least 90% confidence are about 99.5% correct and cover about 81% of all predictions.
- **Disease names help a lot:** when the obvious disease-name words are removed, accuracy drops from 95.2% to 89.4%, so the model learns real medical vocabulary but also relies partly on the names.

**Train vs test check (overfitting):**

| Model | Train accuracy | Test accuracy | Gap |
|---|---|---|---|
| Linear SVM | 99.85% | 95.51% | 4.3 pts |
| Logistic Regression | 99.82% | 95.21% | 4.6 pts |
| Random Forest | 99.93% | 94.08% | 5.9 pts |
| Naive Bayes | 90.11% | 88.22% | 1.9 pts |

Random Forest overfits the most, and Naive Bayes is too simple for the data (low on both). Logistic Regression and Linear SVM learn the pattern well, with a moderate gap that is typical for 50,000 TF-IDF features.

Models are compared by **macro F1** because the classes are imbalanced: it gives small categories like sickle cell anemia the same weight as breast cancer.

## Streamlit App

| Page | What it does |
|---|---|
| **Predict** | Paste a trial summary and get the predicted disease, a confidence score, a probability chart and a low-confidence warning. A second tab classifies a whole CSV and lets you download the results |
| **EDA Dashboard** | Category distribution, top words per category, summary length plots |
| **Model Performance** | Accuracy, precision, recall, F1, per-category scores and confusion matrix |
| **Insights & Reporting** | Written insights on category distribution, medical terms, text patterns, model performance and decision support (with a confidence-threshold review simulator), plus a downloadable report |
| **About** | Project and pipeline description |


## Dataset


- **60,337 clinical trials**, 16 columns (60,064 after removing 273 duplicate summaries)
- **Input:** `brief_summary` (about 102 words on average)
- **Target:** `source_condition_query`, with 8 disease categories

| Disease category | Trials | Share |
|---|---|---|
| breast cancer | 16,301 | 27.0% |
| type 2 diabetes | 11,467 | 19.0% |
| covid-19 | 10,153 | 16.8% |
| anxiety | 9,286 | 15.4% |
| chronic obstructive pulmonary disease | 6,181 | 10.2% |
| rheumatoid arthritis | 3,637 | 6.0% |
| glaucoma | 2,173 | 3.6% |
| sickle cell anemia | 1,139 | 1.9% |

> **Note:** the raw CSV (about 152 MB) and the cleaned CSV (about 109 MB) are larger than GitHub's 100 MB file limit, so they are **not included in this repository**. See [Getting started](#getting-started) to generate them.

## Methodology

1. **Data collection:** load the CSV with Pandas, inspect structure and types, select `brief_summary` as input and the disease label as target.
2. **Preprocessing:** remove missing values and duplicates, lowercase, remove URLs, punctuation and digits, remove stop words, tokenize, lemmatize (NLTK), encode labels, then split 80/20 with stratification.
3. **EDA:** category distribution, frequent words and conditions, summary length, signature words per category, study type and trial status, all with Plotly.
4. **Feature engineering:** TF-IDF with 50,000 features, unigrams and bigrams, `min_df=2`, sublinear term frequency. It is fitted on the training data only, so there is no data leakage.
5. **Modeling:** Logistic Regression, Linear SVM, Naive Bayes and Random Forest, with balanced class weights to handle class imbalance. Models are compared by macro F1. Logistic Regression is the final model: it is within 0.5 points of the best score and gives probabilities directly.
6. **Prediction system:** Logistic Regression returns the predicted category, a confidence score and the top candidates for any new summary.
7. **Streamlit app and insights report.**

## Project structure

```
clinical-trial-disease-classification/
├── app.py                          # Streamlit app
├── retrain_models.py               # rebuilds the model files for your machine
├── requirements.txt
├── README.md
├── notebooks/
│   └── Clinical_Trial_Disease_Category_Classification_Using_NLP_and_ML.ipynb
py
├── data/                           # not in the repo (too large) - see below
├── models/                         # generated by retrain_models.py

```

## Getting started

**Prerequisites:** Python 3.10 or newer.

**1. Clone the repository and install the packages**

```bash
git clone https://github.com/ThereseRansey/clinical-trial-disease-classification.git
cd clinical-trial-disease-classification
pip install -r requirements.txt
```

**2. Add the dataset.** Place the raw file in the `data/` folder:

```
data/clinical_trials_raw_patient2trial_conditions.csv
```

**3. Create the cleaned dataset.** Run either the notebook in `notebooks/`, or the pipeline scripts from the project root:

```bash
python pipeline/01_data_collection.py
python pipeline/02_data_preprocessing.py
```

This creates `data/cleaned_data.csv`.

**4. Build the model files for your machine**

```bash
python retrain_models.py
```

This trains the final model (Logistic Regression, about 30 seconds; set `MODEL = "svm"` at the top of the script to use Linear SVM instead) and saves `tfidf_vectorizer.pkl`, `label_encoder.pkl` and `final_model.pkl` in `models/`. Rebuilding them locally avoids scikit-learn version mismatch errors, because saved model files only load reliably with the version that created them.

**5. Run the app**

```bash
python -m streamlit run app.py
```

It opens at http://localhost:8501.

To explore the full analysis step by step, run the scripts `pipeline/03_eda_plotly.py`, `04_model_training.py` and `05_disease_prediction_system.py`, or open the notebook. They use `# %%` cells, so they run cell by cell in VS Code or Jupyter.

## Limitations and future work

- **One label per trial.** A trial can cover several conditions, but the model must choose one. Mixed-topic summaries get lower confidence.
- **Labels come from search queries,** not expert annotation, so some labels may be noisy.
- **Class imbalance** affects the small categories (glaucoma, sickle cell anemia), even with balanced weights.
- **TF-IDF ignores word order.** Future work: domain-specific language models, more disease categories, multi-label classification, and handling of abbreviations and synonyms with medical vocabularies.
- The model supports human judgment and does not replace it. Low-confidence predictions should be reviewed by a person.

## Tech stack

Python, Pandas, NumPy, NLTK, scikit-learn, Plotly, Streamlit, Joblib

## Author

**Therese Ransey** - [GitHub](https://github.com/ThereseRansey)

