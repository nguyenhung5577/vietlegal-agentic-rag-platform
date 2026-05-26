import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retriever import LegalRetriever


def _contains_any(value: str, expected: list[str]) -> bool:
    normalized = value.lower()
    return any(item.lower() in normalized for item in expected)


def evaluate(golden_path: Path, top_k: int) -> int:
    cases = json.loads(golden_path.read_text(encoding="utf-8"))
    retriever = LegalRetriever()
    rows = []

    for case in cases:
        results = retriever.retrieve(case["query"], expand_links=False)[:top_k]
        joined_numbers = " | ".join(str(r["metadata"].get("number", "")) for r in results)
        joined_articles = " | ".join(str(r["metadata"].get("article", "")) for r in results)

        number_hit = _contains_any(joined_numbers, case.get("expected_numbers", []))
        article_hit = _contains_any(joined_articles, case.get("expected_articles", []))
        passed = number_hit and article_hit
        rows.append((case["id"], passed, number_hit, article_hit, results[0] if results else None))

    passed_count = sum(1 for _, passed, _, _, _ in rows if passed)
    print(f"Retrieval evaluation: {passed_count}/{len(rows)} passed at top_k={top_k}")
    print()

    for case_id, passed, number_hit, article_hit, top in rows:
        status = "PASS" if passed else "FAIL"
        if top:
            top_label = f"{top['metadata'].get('number')} - {top['metadata'].get('article')}"
        else:
            top_label = "NO RESULTS"
        print(f"{status} {case_id}: number_hit={number_hit} article_hit={article_hit}")
        print(f"  top: {top_label}")

    return 0 if passed_count == len(rows) else 1


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval against golden legal queries.")
    parser.add_argument("--golden", default="data/eval/golden_queries.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    raise SystemExit(evaluate(ROOT / args.golden, args.top_k))


if __name__ == "__main__":
    main()
