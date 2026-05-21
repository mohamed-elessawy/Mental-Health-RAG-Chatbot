import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle, os

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
assert os.path.exists("data/train.parquet"), "Preprocessed data not found. Run eda.py first."

label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
df_train = pd.read_parquet("data/train.parquet")
df_val   = pd.read_parquet("data/val.parquet")

# TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(df_train["text"])
X_val   = vectorizer.transform(df_val["text"])

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, df_train["label"])

preds = clf.predict(X_val)
report = classification_report(df_val["label"], preds, target_names=label_names)
print("\n=== Baseline Report (TF-IDF + LogReg) ===")
print(report)

with open("outputs/baseline_report.txt", "w") as f:
    f.write(report)

# Save model for later comparison
pickle.dump({"vectorizer": vectorizer, "clf": clf}, open("models/baseline.pkl", "wb"))
print("Saved → models/baseline.pkl")