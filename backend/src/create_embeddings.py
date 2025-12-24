import pandas as pd
import ast
from sentence_transformers import SentenceTransformer

# Load cleaned data
df = pd.read_csv("data/raw/dailydialog/train.csv")

# Re-apply parsing (same logic as before)
def split_into_turns(dialog_str):
    import re
    turns = re.split(r'[.!?]', dialog_str)
    return [t.strip() for t in turns if t.strip()]

df["dialog"] = df["dialog"].apply(
    lambda x: split_into_turns(ast.literal_eval(x)[0])
)
df["emotion"] = df["emotion"].apply(
    lambda x: list(map(int, x.strip("[]").split()))
)

df["length_match"] = df.apply(
    lambda row: len(row["dialog"]) == len(row["emotion"]),
    axis=1
)
df = df[df["length_match"]].reset_index(drop=True)

# Load sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Take ONE conversation for now
sample_dialog = df.loc[0, "dialog"]

embeddings = model.encode(sample_dialog)

print("Number of turns:", len(sample_dialog))
print("Embedding shape:", embeddings.shape)
