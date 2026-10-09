import json
import unittest
from pathlib import Path
from unittest.mock import patch

from check_citations import check
from research import REPORT_PATH, SOURCES_PATH, save_outputs, slugify, summarize
from tools import RetryableError, with_retry


class RetryTests(unittest.TestCase):
    @patch("tools.time.sleep")
    @patch("tools.random.uniform", return_value=0.0)
    def test_retry_succeeds_and_sleeps(self, _random, sleep):
        calls = []

        def operation():
            calls.append(1)
            if len(calls) < 3:
                raise RetryableError("temporary")
            return "ok"

        self.assertEqual(with_retry(operation, attempts=3, base=2, cap=10), "ok")
        self.assertEqual(sleep.call_args_list[0].args[0], 2)
        self.assertEqual(sleep.call_args_list[1].args[0], 4)

    @patch("tools.time.sleep")
    def test_no_sleep_after_last_failure(self, sleep):
        with self.assertRaises(RetryableError):
            with_retry(lambda: (_ for _ in ()).throw(RetryableError("no")), attempts=1)
        sleep.assert_not_called()


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"n": 1, "url": "https://example.com/a"},
            {"n": 2, "url": "https://example.com/b"},
            {"n": 3, "url": "https://example.com/c"},
        ]

    def test_valid_grouped_citations(self):
        report = """# Survey

Evidence [1, 2] and more [3]. Ignore `code [99]` and [9](https://example.com).

## References
[1] A. web. https://example.com/a (2026-01-01)
[2] B. web. https://example.com/b (2026-01-01)
[3] C. web. https://example.com/c (2026-01-01)
"""
        self.assertEqual(check(report, self.sources), [])

    def test_bad_reference_url_is_reported(self):
        report = """# Survey
Claim [1][2][3].
## References
[1] A https://example.com/wrong
[2] B https://example.com/b
[3] C https://example.com/c
"""
        self.assertTrue(any("[1] URL" in problem for problem in check(report, self.sources)))


class ResearchTests(unittest.TestCase):
    def test_slugify_is_safe(self):
        self.assertEqual(slugify("../../Hello, World!"), "hello-world")
        self.assertEqual(slugify("..."), "topic")
        self.assertLessEqual(len(slugify("x" * 100)), 60)

    def test_summarize_counts_task_calls_and_tokens(self):
        messages = [{
            "tool_calls": [{"name": "task"}, {"name": "task"}, {"name": "write_file"}],
            "usage_metadata": {"input_tokens": 10, "output_tokens": 4},
        }]
        result = summarize(messages, 1.26, "model")
        self.assertEqual(result["subagent_calls"], 2)
        self.assertEqual(result["tokens"], {"input": 10, "output": 4})
        self.assertEqual(result["elapsed_s"], 1.3)

    def test_save_outputs_rejects_missing_report_without_writes(self):
        class Backend:
            def download_files(self, paths):
                class Result:
                    def __init__(self, path, content):
                        self.path, self.content = path, content
                return [Result(REPORT_PATH, None), Result(SOURCES_PATH, b"[]")]

        with patch.object(Path, "write_text") as write_text:
            with self.assertRaises(RuntimeError):
                save_outputs(Backend(), "topic", [], 0, "model", Path("unused"))
            write_text.assert_not_called()

    def test_save_outputs_writes_three_files(self):
        source = {"n": 1, "id": "x", "url": "https://example.com", "title": "X", "date": "2026-01-01", "source": "web"}

        class Backend:
            def download_files(self, paths):
                class Result:
                    def __init__(self, path, content):
                        self.path, self.content = path, content
                return [
                    Result(REPORT_PATH, b"# Report\n"),
                    Result(SOURCES_PATH, json.dumps([source]).encode()),
                ]

        with patch.object(Path, "mkdir"), patch.object(Path, "write_text") as write_text:
            path = save_outputs(Backend(), "My Topic", [], 1, "model", Path("reports-test"))
            self.assertEqual(path.name, "my-topic.md")
            self.assertEqual(write_text.call_count, 3)


if __name__ == "__main__":
    unittest.main()
