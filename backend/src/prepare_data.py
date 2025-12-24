import pandas as pd
import ast
import re
import os

# -------------------------------
# Helper: split dialog into turns
# -------------------------------
def split_into_turns(dialog_str):
    turns = re.split(r"[.!?]", dialog_str)
    return [t.strip() for t in turns if t.strip()]

# -------------------------------
# Load RAW DailyDialog data
# -------------------------------
df = pd.read_csv("data/raw/dailydialog/train.csv")
print("Raw shape:", df.shape)

dialogs = []
emotions = []

# -------------------------------
# Clean & align conversations
# -------------------------------
for _, row in df.iterrows():
    # dialog: ["long sentence ..."]
    raw_dialog = ast.literal_eval(row["dialog"])[0]
    dialog_turns = split_into_turns(raw_dialog)

    # emotion: [0 0 0 4 4]
    emotion_list = list(map(int, row["emotion"].strip("[]").split()))

    if len(dialog_turns) == len(emotion_list) and len(dialog_turns) >= 2:
        dialogs.append(dialog_turns)
        emotions.append(emotion_list)

clean_df = pd.DataFrame({
    "dialog": dialogs,
    "emotion": emotions
})

print("Cleaned shape:", clean_df.shape)

# -------------------------------
# Save CLEAN data
# -------------------------------
os.makedirs("data/processed", exist_ok=True)
clean_df.to_csv("data/processed/clean_dialogs.csv", index=False)

print("✅ Saved to data/processed/clean_dialogs.csv")
