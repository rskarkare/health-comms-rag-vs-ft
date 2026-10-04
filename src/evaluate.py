"""Score each system's answers against the reference answers.

    python src/evaluate.py

Writes results/summary.csv and results/examples.md.
"""
import os

import numpy as np
import pandas as pd
import textstat
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from common import EMBED_MODEL, RESULTS_DIR, get_device, read_jsonl

MODES = ["base", "rag", "finetuned"]
N_EXAMPLES = 5


def reading_grade(texts):
    """Average US school grade level needed to read the texts (lower = plainer)."""
    return float(np.mean([textstat.flesch_kincaid_grade(t) for t in texts if t.strip()]))


def main():
    embedder = SentenceTransformer(EMBED_MODEL, device=get_device())
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    all_predictions, summary = {}, []
    for mode in MODES:
        path = f"{RESULTS_DIR}/predictions_{mode}.jsonl"
        if not os.path.exists(path):
            print(f"Skipping {mode}: {path} not found")
            continue
        rows = read_jsonl(path)
        all_predictions[mode] = rows
        preds = [r["prediction"] for r in rows]
        refs = [r["reference"] for r in rows]

        # Word overlap with the reference answer (0 to 1)
        rouge = np.mean([scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(refs, preds)])
        # Similarity of meaning with the reference answer (cosine, roughly 0 to 1)
        p_vec = embedder.encode(preds, normalize_embeddings=True)
        r_vec = embedder.encode(refs, normalize_embeddings=True)
        similarity = float(np.mean(np.sum(p_vec * r_vec, axis=1)))
        # Did RAG retrieve the document that actually answers the question?
        hit = np.mean([r["id"] in r["retrieved_ids"] for r in rows]) if mode == "rag" else np.nan

        summary.append(
            {
                "system": mode,
                "n": len(rows),
                "rougeL": rouge,
                "semantic_similarity": similarity,
                "reading_grade": reading_grade(preds),
                "avg_words": np.mean([len(p.split()) for p in preds]),
                "retrieval_hit@3": hit,
            }
        )

    if not summary:
        print("No prediction files found. Run src/predict.py first.")
        return

    # The reference answers themselves, for comparing readability and length
    refs = [r["reference"] for r in next(iter(all_predictions.values()))]
    summary.append(
        {
            "system": "reference (MedQuAD)",
            "n": len(refs),
            "rougeL": np.nan,
            "semantic_similarity": np.nan,
            "reading_grade": reading_grade(refs),
            "avg_words": np.mean([len(r.split()) for r in refs]),
            "retrieval_hit@3": np.nan,
        }
    )

    df = pd.DataFrame(summary).round(3)
    print("\n" + df.to_string(index=False))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(f"{RESULTS_DIR}/summary.csv", index=False)

    # Side-by-side answers for a few questions, to read with your own eyes
    by_id = {mode: {r["id"]: r for r in rows} for mode, rows in all_predictions.items()}
    first_rows = next(iter(all_predictions.values()))[:N_EXAMPLES]
    lines = ["# Example answers\n"]
    for row in first_rows:
        lines.append(f"## {row['question']}\n")
        lines.append(f"**Reference:** {row['reference']}\n")
        for mode in by_id:
            if row["id"] in by_id[mode]:
                lines.append(f"**{mode}:** {by_id[mode][row['id']]['prediction']}\n")
        lines.append("---\n")
    with open(f"{RESULTS_DIR}/examples.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\nSaved {RESULTS_DIR}/summary.csv and {RESULTS_DIR}/examples.md")


if __name__ == "__main__":
    main()
