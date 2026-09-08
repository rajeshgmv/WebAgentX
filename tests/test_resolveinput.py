import json
import unittest
from unittest.mock import patch

from src.resolveinput import GeneratedInputResolution, resolve_required_input


class FakeStructuredLlm:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.response


class FakeLlm:
    def __init__(self, response):
        self.structured = FakeStructuredLlm(response)
        self.schema = None
        self.options = None

    def with_structured_output(self, schema, **options):
        self.schema = schema
        self.options = options
        return self.structured


class ResolveInputTests(unittest.TestCase):
    def test_resolves_abbreviation_using_required_field_context(self):
        fake_llm = FakeLlm(
            GeneratedInputResolution(
                resolved_value="Milwaukee",
                interpretation="MKE is Milwaukee in this location field.",
                confidence="high",
                ambiguity=None,
            )
        )
        context = {
            "normalized_objective": "Find a doctor named Sapna",
            "field": {"label": "Near (required)", "type": "text"},
        }

        with patch(
            "src.resolveinput.initialize_groq_llm",
            return_value=fake_llm,
        ):
            result = resolve_required_input(" MKE ", context, api_key="test-key")

        self.assertEqual(result.original_value, "MKE")
        self.assertEqual(result.resolved_value, "Milwaukee")
        self.assertTrue(result.changed)
        self.assertIs(fake_llm.schema, GeneratedInputResolution)
        self.assertEqual(
            fake_llm.options,
            {"method": "function_calling", "include_raw": True},
        )
        payload = json.loads(fake_llm.structured.messages[1][1])
        self.assertEqual(payload["required_input_context"], context)

    def test_rejects_empty_input_before_calling_groq(self):
        with patch("src.resolveinput.initialize_groq_llm") as initialize:
            with self.assertRaisesRegex(ValueError, "cannot be empty"):
                resolve_required_input("  ", {}, api_key="test-key")

        initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
