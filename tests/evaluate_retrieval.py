"""
Simple retrieval evaluation harness.

Usage:
  1. Fill in EVAL_SET below with real questions from YOUR indexed PDFs and the
     source filename you expect each answer to come from.
  2. Run: python tests/evaluate_retrieval.py
  3. It reports retrieval accuracy (was the correct source in the top-k?) and
     average retrieval latency — both are legitimate resume metrics.
"""
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embeddings import query_store

# Fill this in with real question/expected-source pairs from your own PDFs.
EVAL_SET = [
    {"question": "What is the main topic of the document?", "expected_source": "example.pdf"},
    # {"question": "...", "expected_source": "..."},
]


def run_evaluation(top_k: int = 4):
    if not EVAL_SET or EVAL_SET[0]["expected_source"] == "example.pdf":
        print("⚠️  EVAL_SET is still the placeholder. Add real Q&A pairs from your indexed PDFs before running.")
        return

    correct = 0
    latencies = []

    for item in EVAL_SET:
        start = time.perf_counter()
        results = query_store(item["question"], top_k=top_k)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        retrieved_sources = {r["source"] for r in results}
        hit = item["expected_source"] in retrieved_sources
        correct += int(hit)

        print(f"Q: {item['question']}")
        print(f"   expected: {item['expected_source']} | hit: {hit} | latency: {elapsed:.3f}s")

    accuracy = correct / len(EVAL_SET) * 100
    avg_latency = sum(latencies) / len(latencies)

    print("\n--- Results ---")
    print(f"Retrieval accuracy (top-{top_k}): {accuracy:.1f}% ({correct}/{len(EVAL_SET)})")
    print(f"Average retrieval latency: {avg_latency:.3f}s")


if __name__ == "__main__":
    run_evaluation()
