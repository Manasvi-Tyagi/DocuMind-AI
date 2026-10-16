import unittest
from types import SimpleNamespace as N
from unittest.mock import patch
import httpx

from documind.config import Settings, backend_ready, model_name
from documind import ai
from documind.ollama_backend import chat


class OllamaTests(unittest.TestCase):
    @patch("documind.ollama_backend.httpx.Client")
    def test_native_images_json_and_no_auth(self, client):
        client.return_value.__enter__.return_value.post.return_value = httpx.Response(
            200, json={"message": {"content": '{"total":100}'}, "done_reason": "stop"}
        )
        settings = Settings("", "unused", None)
        result = chat(
            settings,
            messages=ai.document_messages(
                "Extract JSON",
                "Page 1",
                [N(number=1, data_url="data:image/jpeg;base64,dGVzdA==")],
            ),
            response_format={"type": "json_object"},
            max_completion_tokens=512,
        )
        args = client.return_value.__enter__.return_value.post.call_args
        self.assertEqual(args.args[0], "http://127.0.0.1:11434/api/chat")
        payload = args.kwargs["json"]
        self.assertEqual(payload["model"], "gemma3:4b")
        self.assertEqual(payload["messages"][1]["images"], ["dGVzdA=="])
        self.assertEqual(payload["format"], "json")
        self.assertFalse(payload["stream"])
        self.assertNotIn("api_key", payload)
        self.assertEqual(result.choices[0].message.content, '{"total":100}')
        self.assertTrue(backend_ready(settings))
        self.assertEqual(model_name(settings, vision=True), "gemma3:4b")

    @patch.object(ai, "Groq")
    @patch.object(ai, "ollama_chat")
    @patch.object(ai, "get_settings", return_value=Settings("", "unused", None))
    def test_local_backend_never_falls_back_to_cloud(self, settings, local, cloud):
        local.return_value = N(
            choices=[N(message=N(content="Invoice"), finish_reason="stop")]
        )
        self.assertEqual(ai.classify_document("Invoice").strip(), "Invoice")
        cloud.assert_not_called()
        self.assertEqual(local.call_args.kwargs["model"], "gemma3:4b")

    @patch("documind.ollama_backend.httpx.Client")
    def test_missing_model_and_connection_have_actionable_errors(self, client):
        settings = Settings("", "unused", None)
        post = client.return_value.__enter__.return_value.post
        post.return_value = httpx.Response(404, json={"error": "not found"})
        with self.assertRaisesRegex(RuntimeError, "ollama pull gemma3:4b"):
            chat(settings, messages=[])
        post.side_effect = httpx.ConnectError("connection refused")
        with self.assertRaisesRegex(RuntimeError, "ollama serve"):
            chat(settings, messages=[])
