from __future__ import annotations

import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from teacherlm_core.retrieval.evaluation import (  # noqa: E402
    RetrievalCase,
    evaluate_case,
    summarize_results,
)


class RetrievalEvaluationTests(unittest.TestCase):
    def test_retrieval_metrics_reward_early_relevant_hits(self) -> None:
        case = RetrievalCase(
            id="q1",
            query="What is collaborative filtering?",
            relevant_chunk_ids={"c2", "c5"},
        )
        result = evaluate_case(
            case,
            retrieved_ids=["c2", "c9", "c5", "c1"],
            k_values=(1, 3),
        )

        self.assertEqual(result.metrics["hit_rate@1"], 1.0)
        self.assertAlmostEqual(result.metrics["precision@3"], 2 / 3)
        self.assertEqual(result.metrics["recall@3"], 1.0)
        self.assertEqual(result.metrics["mrr@3"], 1.0)
        self.assertGreater(result.metrics["ndcg@3"], 0.9)

    def test_summary_reports_failed_cases(self) -> None:
        missed = evaluate_case(
            RetrievalCase(
                id="missed",
                query="Define matrix factorization",
                relevant_chunk_ids={"gold"},
            ),
            retrieved_ids=["other"],
            k_values=(1,),
        )

        summary = summarize_results([missed], k_values=(1,))
        self.assertEqual(summary["case_count"], 1)
        self.assertEqual(summary["metrics"]["hit_rate@1"], 0.0)
        self.assertEqual(summary["failed_cases"][0]["id"], "missed")

    def test_section_recall_and_citation_precision_are_reported(self) -> None:
        case = RetrievalCase(
            id="q2",
            query="Explain SVD",
            relevant_chunk_ids={"c1"},
            relevant_section_ids={"s1"},
            expected_source_document="course.pdf",
        )
        result = evaluate_case(
            case,
            retrieved_ids=["c1", "c2"],
            retrieved_sources=["course.pdf", "other.pdf"],
            retrieved_section_ids=["s1", "s9"],
            cited_ids=["c1", "c9"],
            k_values=(1, 2),
            latency_ms=12.5,
        )

        self.assertEqual(result.metrics["section_recall@1"], 1.0)
        self.assertEqual(result.metrics["citation_precision"], 0.5)
        self.assertEqual(result.metrics["source_document_hit"], 1.0)
        self.assertEqual(result.metrics["latency_ms"], 12.5)


if __name__ == "__main__":
    unittest.main()
