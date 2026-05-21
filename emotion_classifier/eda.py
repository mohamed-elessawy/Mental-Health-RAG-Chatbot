import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

os.makedirs("outputs", exist_ok=True)

# loading dataset
dataset = load_dataset("dair-ai/emotion")
label_names = dataset["train"].features["label"].names  
# the 6 classes --> ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# change dataset to dataframe and add emotion labels and text length
def to_df(split):
    df = pd.DataFrame(dataset[split])
    df.columns = ["text", "label"]
    df["emotion"] = df["label"].map(lambda x: label_names[x])
    df["length"]  = df["text"].apply(lambda x: len(x.split()))
    return df

df_train = to_df("train")
df_val   = to_df("validation")
df_test  = to_df("test")

# preprocessing
def preprocess(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    original_len = len(df)

    # drop duplicates
    df = df.drop_duplicates(subset=["text", "label"])
    dupes_dropped = original_len - len(df)

    # drop null/empty texts
    df = df[df["text"].notna()]
    df = df[df["text"].str.strip() != ""]

    # strip HTML artifacts (e.g. "img src http s ...")
    df["text"] = df["text"].str.replace(r"<[^>]+>", "", regex=True)
    df["text"] = df["text"].str.replace(r"img\s+src\s+\S+", "", regex=True)
    df["text"] = df["text"].str.replace(r"http\S+", "", regex=True)

    # normalize whitespace
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    # remove rows that became empty after cleaning
    df = df[df["text"].str.strip() != ""]

    # drop extreme-length outliers
    # --> too short: ≤ 2 words 
    # --> too long: > 300 words (as the max length in the dataset is 280 -> 300)
    df["length"] = df["text"].apply(lambda x: len(x.split()))
    before_len_filter = len(df)
    df = df[(df["length"] >= 3) & (df["length"] <= 300)]
    length_dropped = before_len_filter - len(df)

    print(f"\n=== Preprocessing — {split_name} ===")
    print(f"  Original rows   : {original_len}")
    print(f"  Duplicates dropped   : {dupes_dropped}")
    print(f"  Length outliers dropped : {length_dropped}")
    print(f"  Final rows      : {len(df)}")

    return df.reset_index(drop=True)


# apply preprocessing 
df_train = preprocess(df_train, "train")
df_val = preprocess(df_val,   "validation")
df_test = preprocess(df_test,  "test")


# save cleaned data for later use (in training all models)
os.makedirs("data", exist_ok=True)

# .parquet for faster to read/write (binary, compressed) 
df_train.to_parquet("data/train.parquet", index=False)
df_val.to_parquet("data/val.parquet",     index=False)
df_test.to_parquet("data/test.parquet",   index=False)
print("\nSaved cleaned data successfully")

# class distribution in training set
print("\n=== Class Distribution (train) ===")
print(df_train["emotion"].value_counts())

df_train["emotion"].value_counts().plot(kind="bar", color="steelblue", edgecolor="white")
plt.title("Emotion class distribution — training set")
plt.ylabel("Samples")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/class_distribution.png")
plt.close()
print("Saved → outputs/class_distribution.png")

# average text length per emotion
df_train.groupby("emotion")["length"].mean().plot(kind="bar", color="coral", edgecolor="white")
plt.title("Average word count per emotion")
plt.ylabel("Words")
plt.tight_layout()
plt.savefig("outputs/length_per_emotion.png")
plt.close()
print("Saved → outputs/length_per_emotion.png")

# printing some sample texts for each emotion
print("\n=== Sample texts per emotion ===")
for emotion in label_names:
    print(f"\n--- {emotion.upper()} ---")
    samples = df_train[df_train["emotion"] == emotion]["text"].sample(3, random_state=42)
    for s in samples:
        print(f"  • {s}")