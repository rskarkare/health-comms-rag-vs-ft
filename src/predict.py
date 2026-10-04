"""Answer the test questions with one of three systems.

    python src/predict.py --mode base        # plain model, no help (the control)
    python src/predict.py --mode rag         # plain model + retrieved documents
    python src/predict.py --mode finetuned   # LoRA fine-tuned model, no documents

Add --limit 5 to try it on just 5 questions first.
"""
import argparse

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from common import (
    ADAPTER_DIR,
    DATA_DIR,
    EMBED_MODEL,
    RESULTS_DIR,
    build_messages,
    generate_answer,
    get_device,
    load_model,
    read_jsonl,
    write_jsonl,
)

TOP_K = 3  # number of documents RAG retrieves per question


class Retriever:
    """Finds the knowledge-base answers whose meaning is closest to a question."""

    def __init__(self):
        self.docs = read_jsonl(f"{DATA_DIR}/knowledge_base.jsonl")
        self.embedder = SentenceTransformer(EMBED_MODEL, device=get_device())
        print(f"Embedding {len(self.docs)} knowledge-base documents...")
        # Each document becomes a vector of 384 numbers. normalize_embeddings=True
        # makes a dot product equal to cosine similarity.
        self.doc_vectors = self.embedder.encode(
            [d["Answer"] for d in self.docs],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, question, k=TOP_K):
        q = self.embedder.encode([question], normalize_embeddings=True)[0]
        # For 10k documents a plain matrix product is exact and instant.
        # FAISS does the same job for millions of documents.
        scores = self.doc_vectors @ q
        best = np.argsort(-scores)[:k]
        return [self.docs[i] for i in best]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "rag", "finetuned"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="answer only the first N questions")
    args = parser.parse_args()

    test = read_jsonl(f"{DATA_DIR}/test.jsonl")
    if args.limit:
        test = test[: args.limit]

    retriever = Retriever() if args.mode == "rag" else None
    model, tokenizer = load_model(ADAPTER_DIR if args.mode == "finetuned" else None)

    results = []
    for row in tqdm(test, desc=f"Answering ({args.mode})"):
        context, retrieved_ids = None, []
        if retriever is not None:
            docs = retriever.search(row["Question"])
            context = "\n\n".join(f"[{i + 1}] {d['Answer']}" for i, d in enumerate(docs))
            retrieved_ids = [d["id"] for d in docs]

        prediction = generate_answer(model, tokenizer, build_messages(row["Question"], context))
        results.append(
            {
                "id": row["id"],
                "qtype": row["qtype"],
                "question": row["Question"],
                "reference": row["Answer"],
                "prediction": prediction,
                "retrieved_ids": retrieved_ids,
            }
        )

    out_path = f"{RESULTS_DIR}/predictions_{args.mode}.jsonl"
    write_jsonl(results, out_path)
    print(f"Saved {len(results)} answers to {out_path}")


if __name__ == "__main__":
    main()
