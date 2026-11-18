"""
Step 6: Streamlit App - Clinical Trial Disease Category Classifier

Run locally:
    streamlit run app.py

Folder layout (next to app.py):
    models/tfidf_vectorizer.pkl
    models/label_encoder.pkl
    models/final_model.pkl
    data/cleaned_data.csv
"""

import re
import socket
import time
from collections import Counter
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Clinical Trial Disease Classifier",
                   page_icon="🧬", layout="wide")

TEMPLATE = "plotly_white"
LOW_CONFIDENCE = 0.70     # below this, predictions were far less accurate in validation

EXAMPLES = {
    "Glaucoma-style summary": "This study evaluates a new eye drop to lower intraocular pressure in patients with open angle disease.",
    "Anxiety-style summary": "A randomized trial of cognitive behavioral therapy to reduce worry and nervousness in college students.",
    "Covid-19-style summary": "We test whether a vaccine booster reduces severe infection caused by the SARS-CoV-2 virus.",
    "Sickle cell-style summary": "Hydroxyurea dosing to reduce pain crises in children with abnormal hemoglobin.",
    "COPD-style summary": "Effect of a bronchodilator inhaler on breathlessness and exacerbations in smokers with airflow limitation.",
    "Mixed topic (ambiguous)": "Depression and anxiety screening in women after breast surgery.",
}


# ---------------------------------------------------------------
# Loading (cached so it only runs once)
# ---------------------------------------------------------------
NLTK_DATA = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def ensure_nltk_data():
    """Download NLTK data only if it is missing (with a timeout, so it can never hang forever)."""
    socket.setdefaulttimeout(20)
    for pkg, path in NLTK_DATA.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[app] downloading NLTK data: {pkg} ...", flush=True)
            if not nltk.download(pkg, quiet=True):
                raise RuntimeError(
                    f"Could not download NLTK data '{pkg}'. Check your internet connection, or run once in a "
                    "terminal: python -c \"import nltk; [nltk.download(p) for p in "
                    "['punkt','punkt_tab','stopwords','wordnet','omw-1.4']]\"")


@st.cache_resource
def load_artifacts():
    ensure_nltk_data()
    start = time.time()
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    print(f"[app] vectorizer loaded in {time.time() - start:.1f}s", flush=True)
    label_encoder = joblib.load("models/label_encoder.pkl")
    model = joblib.load("models/final_model.pkl")
    print(f"[app] model loaded in {time.time() - start:.1f}s", flush=True)
    return tfidf, label_encoder, model, set(stopwords.words("english")), WordNetLemmatizer()


@st.cache_data
def load_data():
    path = Path("data/cleaned_data.csv")
    if not path.exists():
        path = Path("cleaned_data.csv")
    df = pd.read_csv(path)
    df["word_count"] = df["brief_summary"].str.split().str.len()
    return df


try:
    tfidf, label_encoder, model, stop_words, lemmatizer = load_artifacts()
    class_names = label_encoder.classes_
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Put the three .pkl files in a 'models' folder next to app.py.")
    st.stop()
except RuntimeError as e:
    st.error(str(e))
    st.stop()


def preprocess_text(text):
    """Same cleaning pipeline as Step 2."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(lemmatizer.lemmatize(t) for t in tokens)


def predict_proba(texts):
    return model.predict_proba(tfidf.transform([preprocess_text(t) for t in texts]))


@st.cache_resource
def test_results():
    """Re-create the Step 2 test split and score the final model on it."""
    df = load_data()
    y = label_encoder.transform(df["disease_category"])
    _, X_test_text, _, y_test = train_test_split(
        df["processed_text"], y, test_size=0.2, random_state=42, stratify=y)
    X_test = tfidf.transform(X_test_text)
    proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    return y_test, y_pred, proba, X_test


@st.cache_data
def frequent_words():
    """Most frequent words overall and inside each disease category."""
    df = load_data()
    overall, per_cat = Counter(), {}
    for cat, group in df.groupby("disease_category"):
        counts = Counter(" ".join(group["processed_text"]).split())
        per_cat[cat] = counts.most_common(20)
        overall.update(counts)
    return overall.most_common(20), per_cat


# Word stems that name each disease (used to measure how often summaries mention the disease)
NAME_PATTERNS = {
    "anxiety": "anxi",
    "breast cancer": "breast",
    "chronic obstructive pulmonary disease": "copd|pulmonary|obstructive",
    "covid-19": "covid|sars|coronavirus",
    "glaucoma": "glaucoma",
    "rheumatoid arthritis": "rheumatoid|arthritis",
    "sickle cell anemia": "sickle",
    "type 2 diabetes": "diabet",
}


@st.cache_data
def name_mention_rates():
    df = load_data()
    rows = []
    for cat, group in df.groupby("disease_category"):
        pattern = NAME_PATTERNS.get(cat)
        if pattern:
            rows.append({"disease_category": cat,
                         "share": group["processed_text"].str.contains(pattern, regex=True).mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------
st.sidebar.title("🧬 Clinical Trial Classifier")
page = st.sidebar.radio("Go to", ["Predict", "EDA Dashboard", "Model Performance",
                                 "Insights & Reporting"])
st.sidebar.caption("NLP + Machine Learning on clinical trial brief summaries")


# ===============================================================
# PAGE 1: Predict
# ===============================================================
if page == "Predict":
    st.title("Predict the Disease Category")
    st.write("Paste a clinical trial **brief summary** and the model predicts which disease category it belongs to.")

    tab_single, tab_batch = st.tabs(["Single summary", "Batch (CSV upload)"])

    with tab_single:
        st.session_state.setdefault("summary", "")

        def load_example():
            choice = st.session_state.example_choice
            if choice in EXAMPLES:
                st.session_state.summary = EXAMPLES[choice]

        st.selectbox("Load an example (optional)", ["(none)"] + list(EXAMPLES),
                     key="example_choice", on_change=load_example)
        text = st.text_area("Brief summary", key="summary", height=200,
                            placeholder="Paste a clinical trial summary here...")

        if st.button("Predict", type="primary"):
            if len(text.split()) < 5:
                st.warning("Please enter a longer summary (at least 5 words).")
            else:
                proba = predict_proba([text])[0]
                order = proba.argsort()[::-1]
                top_cat, top_conf = class_names[order[0]], proba[order[0]]

                c1, c2 = st.columns(2)
                c1.metric("Predicted category", top_cat)
                c2.metric("Confidence", f"{top_conf:.1%}")

                if top_conf < LOW_CONFIDENCE:
                    st.warning(f"Low confidence (below {LOW_CONFIDENCE:.0%}). In validation, predictions "
                               "this uncertain were often wrong. The text may mix several topics - "
                               "check the second choice below.")
                else:
                    st.success("High confidence prediction.")

                chart = pd.DataFrame({"disease_category": class_names, "probability": proba})
                fig = px.bar(chart, x="probability", y="disease_category", orientation="h",
                             text=chart["probability"].map("{:.1%}".format),
                             title="Probability for each disease category", template=TEMPLATE, height=420)
                fig.update_layout(yaxis={"categoryorder": "total ascending"},
                                  yaxis_title="", xaxis_title="Probability")
                st.plotly_chart(fig)

                with st.expander("See the cleaned text the model used"):
                    st.write(preprocess_text(text))

    with tab_batch:
        st.write("Upload a CSV with a column of clinical trial summaries to classify many at once.")
        file = st.file_uploader("CSV file", type="csv")
        if file is not None:
            up = pd.read_csv(file)
            default = list(up.columns).index("brief_summary") if "brief_summary" in up.columns else 0
            col = st.selectbox("Column that contains the summaries", up.columns, index=default)
            if st.button("Classify all rows"):
                texts = up[col].fillna("").astype(str).tolist()
                with st.spinner(f"Classifying {len(texts):,} summaries..."):
                    proba = predict_proba(texts)
                out = up.copy()
                out["predicted_category"] = class_names[proba.argmax(axis=1)]
                out["confidence"] = proba.max(axis=1).round(4)
                st.dataframe(out.head(100))
                st.plotly_chart(px.histogram(out, y="predicted_category", template=TEMPLATE,
                                             title="Predicted categories"))
                st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                                   "predictions.csv", "text/csv")


# ===============================================================
# PAGE 2: EDA Dashboard
# ===============================================================
elif page == "EDA Dashboard":
    st.title("Exploratory Data Analysis")
    df = load_data()

    c1, c2, c3 = st.columns(3)
    c1.metric("Clinical trials", f"{len(df):,}")
    c2.metric("Disease categories", df["disease_category"].nunique())
    c3.metric("Average summary length", f"{df['word_count'].mean():.0f} words")

    tab1, tab2, tab3 = st.tabs(["Category distribution", "Keywords", "Summary length"])

    with tab1:
        counts = df["disease_category"].value_counts().reset_index()
        counts.columns = ["disease_category", "count"]
        fig = px.bar(counts, x="count", y="disease_category", orientation="h", text="count",
                     title="Number of trials per disease category", template=TEMPLATE, height=450)
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="", xaxis_title="Trials")
        st.plotly_chart(fig)
        st.plotly_chart(px.pie(counts, names="disease_category", values="count", hole=0.45,
                               title="Category share", template=TEMPLATE))

    with tab2:
        choice = st.selectbox("Show the most frequent words for", ["All categories"] + sorted(df["disease_category"].unique()))
        subset = df if choice == "All categories" else df[df["disease_category"] == choice]
        words = Counter(" ".join(subset["processed_text"]).split()).most_common(20)
        wdf = pd.DataFrame(words, columns=["word", "count"])
        fig = px.bar(wdf, x="count", y="word", orientation="h", text="count",
                     title=f"Top 20 words - {choice}", template=TEMPLATE, height=600)
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="", xaxis_title="Frequency")
        st.plotly_chart(fig)

    with tab3:
        st.plotly_chart(px.box(df, x="word_count", y="disease_category", color="disease_category",
                               points=False, title="Summary length by category", template=TEMPLATE,
                               height=500).update_layout(showlegend=False, yaxis_title="",
                                                         xaxis_title="Words per summary"))
        st.plotly_chart(px.histogram(df, x="word_count", nbins=60, marginal="box",
                                     title="Distribution of summary length", template=TEMPLATE))


# ===============================================================
# PAGE 3: Model Performance
# ===============================================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.caption(f"Model: {type(model).__name__} on {len(class_names)} categories - evaluated on the held-out 20% test set.")

    with st.spinner("Evaluating on the test set..."):
        y_test, y_pred, _, _ = test_results()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
    c2.metric("Precision (macro)", f"{precision_score(y_test, y_pred, average='macro'):.1%}")
    c3.metric("Recall (macro)", f"{recall_score(y_test, y_pred, average='macro'):.1%}")
    c4.metric("F1 (macro)", f"{f1_score(y_test, y_pred, average='macro'):.1%}")

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    tab1, tab2, tab3 = st.tabs(["Per-category scores", "Confusion matrix", "Classification report"])
    with tab1:
        pc = pd.DataFrame({
            "disease_category": class_names,
            "precision": [report[c]["precision"] for c in class_names],
            "recall": [report[c]["recall"] for c in class_names],
            "f1": [report[c]["f1-score"] for c in class_names],
        }).melt(id_vars="disease_category", var_name="Metric", value_name="Score")
        fig = px.bar(pc, x="Score", y="disease_category", color="Metric", barmode="group",
                     orientation="h", title="Precision, recall and F1 per category",
                     template=TEMPLATE, height=600)
        fig.update_xaxes(range=[0.5, 1.0])
        fig.update_layout(yaxis_title="")
        st.plotly_chart(fig)
    with tab2:
        cm = confusion_matrix(y_test, y_pred)
        cm_pct = (cm / cm.sum(axis=1, keepdims=True) * 100).round(1)
        fig = px.imshow(cm_pct, x=list(class_names), y=list(class_names), text_auto=True,
                        color_continuous_scale="Blues", aspect="auto",
                        title="Confusion matrix (% of each true category)", template=TEMPLATE, height=650)
        fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig)
    with tab3:
        st.dataframe(pd.DataFrame(report).T.round(3))


# ===============================================================
# PAGE 4: Insights & Reporting  (Step 7)
# ===============================================================
elif page == "Insights & Reporting":
    st.title("Insights & Reporting")
    # st.write("Key findings from the clinical trial data and the trained model. "
    #          "Every number is calculated from your data, so it stays correct if the data or model changes.")

    df = load_data()
    report = ["# Clinical Trial Disease Classification - Insights Report"]

    def add_insights(title, points):
        """Show the key insights as bullets and keep them for the downloadable report."""
        st.markdown("**Key insights**")
        for p in points:
            st.markdown(f"- {p}")
        report.append(f"## {title}\n" + "\n".join(f"- {p.replace('**', '')}" for p in points))

    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Category distribution", "2. Medical terms", "3. Text patterns",
        "4. Model performance"])

    # ---------- 1. Disease category distribution ----------
    with tab1:
        counts = df["disease_category"].value_counts()
        share = counts / counts.sum() * 100
        add_insights("Disease category distribution", [
            f"**{counts.index[0]}** is the largest category with {counts.iloc[0]:,} trials ({share.iloc[0]:.1f}%).",
            f"**{counts.index[-1]}** is the smallest with {counts.iloc[-1]:,} trials ({share.iloc[-1]:.1f}%), "
            f"about {counts.iloc[0] / counts.iloc[-1]:.0f} times fewer than the largest.",
            f"The three largest categories together make up {share.iloc[:3].sum():.0f}% of all {len(df):,} trials.",
            "The classes are imbalanced, so the model was trained with balanced class weights and judged "
            "with macro-averaged metrics, which give small categories equal weight.",
        ])
        cdf = counts.reset_index()
        cdf.columns = ["disease_category", "count"]
        c1, c2 = st.columns(2)
        fig = px.bar(cdf, x="count", y="disease_category", orientation="h", text="count",
                     title="Trials per category", template=TEMPLATE, height=420)
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="", xaxis_title="Trials")
        c1.plotly_chart(fig)
        c2.plotly_chart(px.pie(cdf, names="disease_category", values="count", hole=0.45,
                               title="Share of all trials", template=TEMPLATE, height=420)
                        .update_layout(showlegend=False)
                        .update_traces(textinfo="percent"))

    # ---------- 2. Frequently occurring medical terms ----------
    with tab2:
        top_overall, per_cat = frequent_words()
        in_every_category = set.intersection(*[{w for w, _ in v} for v in per_cat.values()])
        generic = [w for w, _ in top_overall if w in in_every_category][:6]

        y_test, y_pred, proba, X_test = test_results()
        terms = tfidf.get_feature_names_out()
        overall_mean = np.asarray(X_test.mean(axis=0)).ravel()
        sig_rows = []
        for k, cat in enumerate(class_names):
            cat_mean = np.asarray(X_test[y_test == k].mean(axis=0)).ravel()
            # words used MORE in this category than in the data overall
            distinct = cat_mean - overall_mean
            sig_rows.append({"disease_category": cat,
                             "signature terms": ", ".join(terms[distinct.argsort()[::-1][:6]])})
        sig = pd.DataFrame(sig_rows)
        example = sig[sig["disease_category"] == "glaucoma"]
        example = (example if len(example) else sig).iloc[0]

        points = [f"The most frequent words overall are: {', '.join(w for w, _ in top_overall[:8])}."]
        if generic:
            points.append(f"{len(in_every_category)} words appear in the top 20 of every category "
                          f"(for example {', '.join(generic)}). Because every disease uses them, "
                          "they say little about which disease a trial is about.")
        points.append(f"Each disease has its own signature vocabulary, for example "
                      f"**{example['disease_category']}**: {example['signature terms']}. "
                      "This is why a text classifier works well on this data.")
        add_insights("Frequently occurring medical terms", points)

        wdf = pd.DataFrame(top_overall, columns=["word", "count"])
        fig = px.bar(wdf, x="count", y="word", orientation="h", text="count",
                     title="Top 20 most frequent words (all trials)", template=TEMPLATE, height=560)
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="", xaxis_title="Frequency")
        st.plotly_chart(fig)

        st.subheader("Signature terms per disease")
        st.caption("Words used more in this category than in the data overall (average TF-IDF weight above the overall average, test set).")
        st.dataframe(sig, hide_index=True)

    # ---------- 3. Clinical trial text patterns and trends ----------
    with tab3:
        lens = df.groupby("disease_category")["word_count"].mean().sort_values(ascending=False)
        mention = name_mention_rates().sort_values("share", ascending=False)
        cleaned_len = df["processed_text"].str.split().str.len().mean()
        points = [
            f"Summaries average {df['word_count'].mean():.0f} words (median {df['word_count'].median():.0f}); "
            f"after cleaning they shrink to about {cleaned_len:.0f} words "
            f"({(1 - cleaned_len / df['word_count'].mean()):.0%} shorter).",
            f"**{lens.index[0]}** summaries are the longest ({lens.iloc[0]:.0f} words on average) and "
            f"**{lens.index[-1]}** the shortest ({lens.iloc[-1]:.0f}).",
        ]
        if len(mention):
            points.append(
                f"On average {mention['share'].mean():.0%} of summaries contain the disease name or a close term, "
                f"from {mention['share'].min():.0%} (**{mention.iloc[-1]['disease_category']}**) to "
                f"{mention['share'].max():.0%} (**{mention.iloc[0]['disease_category']}**).")
            points.append("This is why the classifier partly depends on disease names. In the Step 4 experiment, "
                          "removing them lowered accuracy by about 6 percentage points, so the model still "
                          "learns real medical vocabulary beyond the names.")
        add_insights("Clinical trial text patterns and trends", points)

        c1, c2 = st.columns(2)
        ldf = lens.reset_index()
        ldf.columns = ["disease_category", "avg_words"]
        fig = px.bar(ldf, x="avg_words", y="disease_category", orientation="h",
                     text=ldf["avg_words"].round(0), title="Average summary length (words)",
                     template=TEMPLATE, height=420)
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="", xaxis_title="Words")
        c1.plotly_chart(fig)
        if len(mention):
            fig = px.bar(mention, x="share", y="disease_category", orientation="h",
                         text=mention["share"].map("{:.0%}".format),
                         title="Summaries that mention the disease name", template=TEMPLATE, height=420)
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="",
                              xaxis_title="Share of summaries", xaxis_tickformat=".0%")
            c2.plotly_chart(fig)

    # ---------- 4. Model performance and prediction accuracy ----------
    with tab4:
        y_test, y_pred, proba, X_test = test_results()
        conf = proba.max(axis=1)
        correct = y_pred == y_test
        high = conf >= 0.90
        rep = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        f1_by_cat = pd.Series({c: rep[c]["f1-score"] for c in class_names}).sort_values()

        cm = confusion_matrix(y_test, y_pred)
        off = cm.copy()
        np.fill_diagonal(off, 0)
        t_idx, p_idx = np.unravel_index(off.argmax(), off.shape)

        add_insights("Model performance and prediction accuracy", [
            f"The final model reaches {accuracy_score(y_test, y_pred):.1%} accuracy and "
            f"{f1_score(y_test, y_pred, average='macro'):.1%} macro F1 on {len(y_test):,} unseen test trials.",
            f"The strongest category is **{f1_by_cat.index[-1]}** (F1 {f1_by_cat.iloc[-1]:.2f}); "
            f"the weakest is **{f1_by_cat.index[0]}** (F1 {f1_by_cat.iloc[0]:.2f}).",
            f"The most common mistake: **{class_names[t_idx]}** trials predicted as "
            f"**{class_names[p_idx]}** ({off[t_idx, p_idx]} cases).",
            f"{high.mean():.0%} of predictions have at least 90% confidence and {correct[high].mean():.1%} of "
            f"those are correct, compared with {correct[~high].mean():.1%} for the rest. "
            "Confidence is a reliable signal of when to trust the model.",
        ])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
        c2.metric("Precision (macro)", f"{precision_score(y_test, y_pred, average='macro'):.1%}")
        c3.metric("Recall (macro)", f"{recall_score(y_test, y_pred, average='macro'):.1%}")
        c4.metric("F1 (macro)", f"{f1_score(y_test, y_pred, average='macro'):.1%}")

        c1, c2 = st.columns(2)
        fdf = f1_by_cat.reset_index()
        fdf.columns = ["disease_category", "f1"]
        fig = px.bar(fdf, x="f1", y="disease_category", orientation="h", text=fdf["f1"].round(2),
                     title="F1 score per category", template=TEMPLATE, height=420)
        fig.update_xaxes(range=[0.5, 1.0])
        fig.update_layout(yaxis_title="", xaxis_title="F1")
        c1.plotly_chart(fig)

        buckets = pd.cut(conf, bins=[0, 0.5, 0.7, 0.9, 1.0001], right=False,
                         labels=["below 50%", "50-70%", "70-90%", "90-100%"])
        rel = (pd.DataFrame({"Confidence": buckets, "Correct": correct})
                 .groupby("Confidence", observed=True)["Correct"].agg(["mean", "count"]).reset_index())
        rel.columns = ["Confidence level", "Accuracy", "Predictions"]
        fig = px.bar(rel, x="Confidence level", y="Accuracy", text=rel["Accuracy"].map("{:.0%}".format),
                     hover_data=["Predictions"], title="Accuracy by confidence level",
                     template=TEMPLATE, height=420)
        fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
        c2.plotly_chart(fig)

#     # ---------- 5. Support for healthcare analytics and research decisions ----------
#     with tab5:
#         y_test, y_pred, proba, X_test = test_results()
#         conf = proba.max(axis=1)
#         correct = y_pred == y_test

#         auto90 = conf >= 0.90
#         add_insights("Support for healthcare analytics and medical research decision-making", [
#             "The model can automatically sort large collections of trial summaries into disease categories, "
#             "so researchers find disease-specific studies faster.",
#             f"At a 90% confidence threshold, {auto90.mean():.0%} of trials can be classified automatically "
#             f"with {correct[auto90].mean():.1%} accuracy, and only {(~auto90).mean():.0%} need a person to check them.",
#             "Category counts can feed analytics dashboards on which disease areas appear most often in a trial "
#             "collection (these counts describe this dataset, not worldwide research activity).",
#             "The model supports human judgment and does not replace it: trials that mix several conditions "
#             "are the most likely to be uncertain, and the labels come from search queries, not expert annotation.",
#         ])

#         st.subheader("Try a review workflow")
#         st.write("Choose how confident the model must be before a prediction is accepted automatically. "
#                  "Everything below it goes to a human reviewer.")
#         threshold = st.slider("Confidence threshold", 0.50, 0.99, 0.90, 0.01, format="%.2f")
#         auto = conf >= threshold

#         c1, c2, c3 = st.columns(3)
#         c1.metric("Classified automatically", f"{auto.mean():.1%}")
#         c2.metric("Accuracy of those predictions", f"{correct[auto].mean():.1%}" if auto.any() else "n/a")
#         c3.metric("Sent for manual review", f"{(~auto).mean():.1%}")

#         flow = pd.DataFrame({
#             "Group": ["Automatic", "Manual review"],
#             "Trials": [int(auto.sum()), int((~auto).sum())],
#             "Model accuracy": [correct[auto].mean() if auto.any() else 0,
#                                correct[~auto].mean() if (~auto).any() else 0],
#         })
#         fig = px.bar(flow, x="Group", y="Trials", color="Group", text="Trials",
#                      hover_data={"Model accuracy": ":.1%"},
#                      title=f"Test trials at a {threshold:.0%} confidence threshold",
#                      template=TEMPLATE, height=380)
#         fig.update_layout(showlegend=False, xaxis_title="")
#         st.plotly_chart(fig)

#         st.subheader("How this supports decisions")
#         st.markdown("""
# - **Organize research records:** group thousands of trial summaries by disease without reading each one.
# - **Focus expert time:** reviewers check only the uncertain predictions instead of every record.
# - **Monitor a trial collection:** track which disease areas are well represented and which are small.
# - **Speed up literature screening:** find the trials relevant to one disease quickly.
# """)

#     st.divider()
#     st.download_button("Download insights report (Markdown)", "\n\n".join(report).encode("utf-8"),
#                        "insights_report.md", "text/markdown")


# ===============================================================
# PAGE 5: About
# ===============================================================
# else:
#     st.title("About this project")
#     st.markdown("""
# **Clinical Trial Disease Category Classification Using NLP and Machine Learning**

# This app reads the *Brief Summary* of a clinical trial and predicts which of 8 disease categories it belongs to.

# **Pipeline**
# 1. Data collection: clinical trials CSV loaded with Pandas
# 2. Preprocessing: duplicates removed, text cleaned, stop words removed, tokenization, lemmatization
# 3. Features: TF-IDF vectorization (words and two-word phrases)
# 4. Models compared: Logistic Regression, Linear SVM, Naive Bayes, Random Forest
# 5. Final model: Logistic Regression, whose predicted probabilities are used as the confidence score
# 6. This Streamlit app for prediction and visualization

# **Tools:** Python, Pandas, NLTK, scikit-learn, Plotly, Streamlit
# """)
#     st.write("**Categories:**", ", ".join(class_names))
