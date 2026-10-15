"""Offline regressions for the user to run; never contact Ollama."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from experiments.checks import check_task, rouge_f1, same_typed_value
from experiments.client import Ollama
from experiments.dataset import TASKS
from experiments.runner import build_plan


class QualityChecks(unittest.TestCase):
    def test_json_format_and_content(self):
        item = next(x for x in TASKS if x["id"] == "json1")
        self.assertTrue(check_task(item, item["reference"])["task_pass"])
        self.assertFalse(check_task(item, "```json\n" + item["reference"] + "\n```")["task_pass"])

    def test_json_types_and_duplicate_keys(self):
        item = dict(check="json", expected={"n": 1})
        self.assertFalse(check_task(item, '{"n":true}')["task_pass"])
        self.assertFalse(check_task(item, '{"n":0,"n":1}')["task_pass"])
        self.assertFalse(same_typed_value({"n": "1"}, {"n": 1}))

    def test_exact_does_not_match_substrings(self):
        item = dict(check="exact", reference="9")
        self.assertFalse(check_task(item, "19")["task_pass"])
        self.assertTrue(check_task(item, "9")["task_pass"])

    def test_rouge_f1_and_punctuation(self):
        self.assertEqual(rouge_f1("Привет, мир!", "привет мир")["rougeL_f1"], 100)
        self.assertAlmostEqual(rouge_f1("a b c", "a")["rouge1_f1"], 50)

    def test_code_is_never_executed_by_default(self):
        item = next(x for x in TASKS if x["id"] == "code1")
        result = check_task(item, item["reference"])
        self.assertIsNone(result["task_pass"])
        self.assertEqual(result["evaluation_status"], "code_not_executed")

    def test_code_fences_fail_format(self):
        item = next(x for x in TASKS if x["id"] == "code1")
        self.assertFalse(check_task(item, "```python\n" + item["reference"] + "\n```")["format_pass"])

    def test_dataset_balance(self):
        self.assertEqual(len(TASKS), 30)
        self.assertEqual(len({t["id"] for t in TASKS}), 30)
        self.assertEqual(len({t["category"] for t in TASKS}), 10)


class Planning(unittest.TestCase):
    def test_unique_plan_and_counterbalanced_order(self):
        args = SimpleNamespace(repeats=2, models=["A", "B"], suite="quality",
            temperatures=[0.0, .7], seed=1729, context=2048, budget=384)
        jobs = build_plan(args)
        self.assertEqual(len(jobs), 240)
        self.assertEqual(len({j["job_id"] for j in jobs}), 240)
        self.assertEqual(jobs, build_plan(args))
        self.assertEqual(jobs[120]["model"], "B")


class Streaming(unittest.TestCase):
    def client_with_events(self, events):
        import json
        client = Ollama()
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.iter_lines.return_value = [json.dumps(e).encode() for e in events]
        client.http = Mock()
        client.http.post.return_value = response
        return client

    def test_exact_counter(self):
        client = self.client_with_events([
            dict(message=dict(content="Привет"), done=False),
            dict(message=dict(content=""), done=True, eval_count=2, eval_duration=20000000)])
        row = client.chat("A", "test", {})
        self.assertEqual(row["tokens"], 2)
        self.assertEqual(row["token_count_source"], "ollama.eval_count")
        self.assertEqual(row["server_tpot_ms"], 10)

    def test_missing_usage_is_not_estimated(self):
        client = self.client_with_events([dict(message=dict(content="text"), done=True)])
        with self.assertRaisesRegex(RuntimeError, "eval_count"):
            client.chat("A", "test", {})

    def test_incomplete_stream_is_an_error(self):
        client = self.client_with_events([dict(message=dict(content="text"), done=False)])
        with self.assertRaisesRegex(RuntimeError, "done=true"):
            client.chat("A", "test", {})


if __name__ == "__main__":
    unittest.main()
