"""
retrain_models.py - rebuild the three model files with the scikit-learn version
installed on THIS computer, so the Streamlit app can load them.

Run once from the project folder (next to app.py):
    python retrain_models.py

Choose the model with the MODEL setting below ("logreg" or "svm").

Needs:   data/cleaned_data.csv
Creates: models/tfidf_vectorizer.pkl, models/label_encoder.pkl, models/final_model.pkl
Takes about 1-2 minutes.
"""

import os
import time

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

MODEL = "logreg"     # "logreg" = Logistic Regression (final choice)
                     # "svm"    = Linear SVM with calibrated probabilities

start = time.time()
print("scikit-learn version on this computer:", sklearn.__version__)

# 1. Load the cleaned data (created in Step 2)
df = pd.read_csv("data/cleaned_data.csv")
df = df.dropna(subset=["processed_text", "disease_category"])
print("Rows:", len(df))

# 2. Encode labels (same alphabetical 0-7 order as before)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["disease_category"])

# 3. Same 80/20 stratified split as Steps 2, 4 and 5
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["processed_text"], y, test_size=0.2, random_state=42, stratify=y)

# 4. TF-IDF with the tuned settings from Step 4
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                        min_df=2, sublinear_tf=True)
X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)
print("TF-IDF matrix:", X_train.shape)

# 5. Train the final model (both options give probabilities for the confidence %)
if MODEL == "logreg":
    print("Training Logistic Regression...")
    model = LogisticRegression(C=10, max_iter=2000, class_weight="balanced", random_state=42)
else:
    print("Training Linear SVM with calibrated confidence scores...")
    model = CalibratedClassifierCV(
        LinearSVC(C=1, class_weight="balanced", random_state=42), cv=5)
model.fit(X_train, y_train)

# 6. Check the result on the test set
y_pred = model.predict(X_test)
print(f"Test accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Test F1 (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")

# 7. Save
os.makedirs("models", exist_ok=True)
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(model, "models/final_model.pkl")
print(f"\nSaved 3 files in models/ ({time.time() - start:.0f}s). Now restart the app:")
print("    python -m streamlit run app.py")
