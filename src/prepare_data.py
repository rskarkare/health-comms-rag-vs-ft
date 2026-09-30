"""Clean MedQuAD and split it into test, fine-tuning, and RAG knowledge-base files."""
import os
from datasets import load_dataset

SEED = 42          # fixed random seed so the shuffle is the same every run
N_TEST = 200       # questions held out for the final comparison
N_TRAIN = 2000     # question-answer pairs for fine-tuning

# 1. Load the data as a pandas table
df = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train").to_pandas()
print("Rows loaded:", len(df))

# 2. Clean the text
df["Question"] = df["Question"].str.replace(r"\s*\?\s*\?\s*$", "?", regex=True).str.strip()
df["Answer"] = df["Answer"].str.split().str.join(" ")    # collapse extra spaces and newlines
df = df.dropna()
df = df.drop_duplicates(subset="Question")
df = df.drop_duplicates(subset="Answer")
df = df[df["Answer"].str.len().between(50, 1500)]       # skip tiny and very long answers
print("Rows after cleaning:", len(df))

# 3. Shuffle, then split
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df["id"] = df.index
test = df.iloc[:N_TEST]
train = df.iloc[N_TEST:N_TEST + N_TRAIN]
knowledge = df[["id", "qtype", "Answer"]]   # every answer, but no questions

# 4. Save as JSON Lines files (one record per line)
os.makedirs("data/processed", exist_ok=True)
test.to_json("data/processed/test.jsonl", orient="records", lines=True)
train.to_json("data/processed/train.jsonl", orient="records", lines=True)
knowledge.to_json("data/processed/knowledge_base.jsonl", orient="records", lines=True)

print(f"Test: {len(test)} | Train: {len(train)} | Knowledge base: {len(knowledge)}")
print("Example cleaned question:", test.iloc[0]["Question"])
