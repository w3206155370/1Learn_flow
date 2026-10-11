import importlib.util
import sys
import unittest
import types
from datetime import date, timedelta
from pathlib import Path

# Core helpers are in app.py; Streamlit is required when running this suite.
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
spec = importlib.util.spec_from_file_location("learnflow_app", APP_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
try:
    import streamlit  # noqa: F401
except ModuleNotFoundError:
    # Core-function tests can run without Streamlit installed; UI execution still requires it.
    sys.modules["streamlit"] = types.SimpleNamespace()
try:
    from openai import OpenAI  # noqa: F401
except ModuleNotFoundError:
    # API calls are not used by these pure-function tests.
    class OpenAI:  # type: ignore[no-redef]
        pass
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=OpenAI)
spec.loader.exec_module(module)


class LearnFlowCoreTests(unittest.TestCase):
    def test_extract_json_plain(self):
        self.assertEqual(module.extract_json('{"score": 88}'), {"score": 88})

    def test_extract_json_fenced(self):
        self.assertEqual(module.extract_json('```json\n{"score": 88}\n```'), {"score": 88})

    def test_schedule_review_remembered(self):
        item = {"question": "q", "review_count": 0, "mastery": 1}
        result = module.schedule_review(item, True)
        self.assertEqual(result["next_review"], (date.today() + timedelta(days=3)).isoformat())
        self.assertEqual(result["mastery"], 2)

    def test_schedule_review_not_remembered(self):
        item = {"question": "q", "review_count": 0, "mastery": 1}
        result = module.schedule_review(item, False)
        self.assertEqual(result["next_review"], (date.today() + timedelta(days=1)).isoformat())
        self.assertEqual(result["mastery"], 0)

    def test_normalize_tasks(self):
        tasks = module.normalize_tasks(["复述概念", "", "写小例子"], "mat-1", "示例")
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["status"], "pending")
        self.assertEqual(tasks[0]["source_material_id"], "mat-1")


if __name__ == "__main__":
    unittest.main()
