import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from documind.config import get_settings, use_provider


class ProviderSelectionTests(unittest.TestCase):
    def test_nested_provider_overrides_restore_the_previous_selection(self):
        default = get_settings().ai_provider
        with use_provider("groq"):
            self.assertEqual(get_settings().ai_provider, "groq")
            with use_provider("ollama"):
                self.assertEqual(get_settings().ai_provider, "ollama")
            self.assertEqual(get_settings().ai_provider, "groq")
        self.assertEqual(get_settings().ai_provider, default)

    @patch("documind.config.backend_ready", return_value=True)
    def test_sidebar_selection_routes_questions_and_clears_old_answers(self, ready):
        def answer(*args, **kwargs):
            return "Answered by " + get_settings().ai_provider

        with patch("documind.ai.answer_question", side_effect=answer):
            app = AppTest.from_file(
                str(Path(__file__).resolve().parent.parent / "app.py")
            ).run(timeout=25)
            app.session_state["document_text"] = "Invoice total 220"
            app.session_state["document_category"] = "Invoice"
            app.run(timeout=25)
            self.assertFalse(app.exception)
            radio = next(r for r in app.radio if r.label == "AI backend")
            radio.set_value("groq").run(timeout=25)
            next(t for t in app.text_input if t.label == "Your question").set_value(
                "What is the total?"
            )
            next(b for b in app.button if b.label == "Ask document").click().run(
                timeout=25
            )
            self.assertFalse(app.exception)
            self.assertEqual(
                app.session_state["messages"][-1]["content"], "Answered by groq"
            )
            next(r for r in app.radio if r.label == "AI backend").set_value(
                "ollama"
            ).run(timeout=25)
            self.assertNotIn("messages", app.session_state)
            next(t for t in app.text_input if t.label == "Your question").set_value(
                "What is the total?"
            )
            next(b for b in app.button if b.label == "Ask document").click().run(
                timeout=25
            )
            self.assertFalse(app.exception)
            self.assertEqual(
                app.session_state["messages"][-1]["content"], "Answered by ollama"
            )
