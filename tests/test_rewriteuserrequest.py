import unittest
from unittest.mock import patch

from src.rewriteuserrequest import (
    ProvidedInput,
    RewrittenUserRequest,
    rewrite_user_request,
)


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


class RewriteUserRequestTests(unittest.TestCase):
    def test_preserves_distinct_name_and_location_inputs(self):
        response = RewrittenUserRequest(
            normalized_request=(
                "Locate a doctor named Sapna in Milwaukee and return the details"
            ),
            ambiguities=[],
            provided_inputs=[
                ProvidedInput(label="person name", value="Sapna"),
                ProvidedInput(label="location", value="Milwaukee"),
            ],
        )
        fake_llm = FakeLlm(response)

        with patch(
            "src.rewriteuserrequest.initialize_groq_llm",
            return_value=fake_llm,
        ):
            result = rewrite_user_request(
                "find dr Sapna and give me details. location is Milwaukee",
                api_key="test-key",
            )

        self.assertEqual(
            [item.model_dump() for item in result.provided_inputs],
            [
                {"label": "person name", "value": "Sapna"},
                {"label": "location", "value": "Milwaukee"},
            ],
        )

    def test_rewrites_request_with_strict_structured_output(self):
        response = RewrittenUserRequest(
            normalized_request=(
                "Locate a doctor named Swapna and return the available public details"
            ),
            ambiguities=["Multiple doctors named Swapna may exist"],
            provided_inputs=[
                ProvidedInput(label="person name", value="Swapna")
            ],
        )
        fake_llm = FakeLlm(response)

        with patch(
            "src.rewriteuserrequest.initialize_groq_llm",
            return_value=fake_llm,
        ) as initialize:
            result = rewrite_user_request(
                " search dr Swapna and provide details ",
                api_key="test-key",
            )

        initialize.assert_called_once_with(
            api_key="test-key",
            temperature=0.0,
            max_tokens=768,
        )
        self.assertEqual(fake_llm.schema, RewrittenUserRequest)
        self.assertEqual(
            fake_llm.options,
            {"method": "function_calling", "include_raw": True},
        )
        self.assertIn("Locate a doctor named Swapna", result.normalized_request)
        self.assertEqual(result.provided_inputs[0].value, "Swapna")
        self.assertEqual(fake_llm.structured.messages[0][0], "system")
        self.assertIn(
            "search dr Swapna and provide details",
            fake_llm.structured.messages[1][1],
        )

    def test_rejects_empty_request_before_initializing_llm(self):
        with patch("src.rewriteuserrequest.initialize_groq_llm") as initialize:
            with self.assertRaisesRegex(ValueError, "cannot be empty"):
                rewrite_user_request("   ", api_key="test-key")

        initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
