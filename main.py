import string
import re
import pandas as pd
import nltk

from antm import ANTM

nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("wordnet")


def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub("[0-9]+", "", text)
    return text


def count_remover(text, threshold=4):
    if len(text.split()) < threshold:
        return pd.NaT
    else:
        return text


data = pd.read_json("./data/tweets.json", lines=True)
df = data[["Text", "CreatedAt"]].rename(
    columns={"Text": "content", "CreatedAt": "time"}
)
df = df.sample(frac=0.05)

df["content"] = df["content"].str.replace(r"@\w+", "")
df["content"] = df["content"].apply(lambda x: re.sub(r"http\S+", "", x))
df["content"] = df["content"].apply(lambda x: remove_punct(x))
df["content"] = df["content"].apply(lambda x: count_remover(x))
df = df.dropna()
df["time"] = pd.to_datetime(df["time"], format="%B %d, %Y at %I:%M%p")
df["time"] = df["time"].dt.to_period("M")
df["time"] = df["time"].apply(lambda x: x.ordinal)

df = df.sort_values("time")
df = df.dropna()
df = df.reset_index(drop=True)
df = df[["content", "time"]]
df = df.reset_index()

window_size = 6
overlap = 2

model = ANTM(
    df,
    overlap,
    window_size,
    mode="data2vec",
    num_words=5,
    path="./saved_data",
)

model.fit(save=True)
