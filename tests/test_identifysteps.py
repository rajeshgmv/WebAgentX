import unittest
from unittest.mock import patch

from pydantic import ValidationError

from src.identifysteps import (
    GeneratedStepPlan,
    Step,
    identify_steps,
    is_valid_http_url,
)
from src.rewriteuserrequest import ProvidedInput, RewrittenUserRequest


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


class IdentifyStepsTests(unittest.TestCase):
    def test_generates_validated_plan_and_preserves_user_input(self):
        generated = GeneratedStepPlan(
            steps=[
                Step(step_number=1, description="Navigate to the homepage"),
                Step(step_number=2, description="Extract the requested result"),
            ],
            success_criteria="The requested result is extracted",
        )
        fake_llm = FakeLlm(generated)

        rewritten = RewrittenUserRequest(
            normalized_request="Locate the requested pricing information",
            ambiguities=[],
            provided_inputs=[
                ProvidedInput(label="item", value="pricing")
            ],
        )
        with (
            patch(
                "src.identifysteps.rewrite_user_request",
                return_value=rewritten,
            ) as rewrite,
            patch(
                "src.identifysteps.initialize_groq_llm",
                return_value=fake_llm,
            ) as initialize,
        ):
            result = identify_steps(
                " https://example.com ",
                " Find pricing ",
                api_key="test-key",
            )

        rewrite.assert_called_once_with("Find pricing", api_key="test-key")
        initialize.assert_called_once_with(
            api_key="test-key",
            temperature=0.0,
            max_tokens=1_200,
        )
        self.assertEqual(fake_llm.schema, GeneratedStepPlan)
        self.assertEqual(
            fake_llm.options,
            {"method": "function_calling", "include_raw": True},
        )
        self.assertEqual(result.url, "https://example.com")
        self.assertEqual(result.user_request, "Find pricing")
        self.assertEqual(
            result.normalized_request,
            "Locate the requested pricing information",
        )
        self.assertEqual(result.ambiguities, [])
        self.assertEqual(result.provided_inputs[0].value, "pricing")
        self.assertEqual(fake_llm.structured.messages[0][0], "system")
        self.assertEqual(fake_llm.structured.messages[1][0], "human")
        planner_input = fake_llm.structured.messages[1][1]
        self.assertIn("Locate the requested pricing information", planner_input)
        self.assertIn("'value': 'pricing'", planner_input)
        self.assertNotIn("Find pricing", planner_input)

    def test_rejects_invalid_url_before_initializing_llm(self):
        with (
            patch("src.identifysteps.rewrite_user_request") as rewrite,
            patch("src.identifysteps.initialize_groq_llm") as initialize,
        ):
            with self.assertRaisesRegex(ValueError, "valid HTTP or HTTPS"):
                identify_steps("https://", "Find pricing", api_key="test-key")

        rewrite.assert_not_called()
        initialize.assert_not_called()

    def test_rejects_non_sequential_steps(self):
        with self.assertRaisesRegex(ValidationError, "sequential"):
            GeneratedStepPlan(
                steps=[Step(step_number=2, description="Extract a result")],
                success_criteria="A result is extracted",
            )

    def test_forbids_unexpected_model_fields(self):
        with self.assertRaises(ValidationError):
            GeneratedStepPlan.model_validate(
                {
                    "steps": [
                        {
                            "step_number": 1,
                            "description": "Navigate to the homepage",
                        }
                    ],
                    "success_criteria": "The page is open",
                    "unexpected": True,
                }
            )

    def test_url_validation(self):
        self.assertTrue(is_valid_http_url("https://example.com/path"))
        self.assertTrue(is_valid_http_url("http://localhost:8501"))
        self.assertFalse(is_valid_http_url("https://"))
        self.assertFalse(is_valid_http_url("javascript:alert(1)"))
        self.assertFalse(is_valid_http_url("https://example.com/has space"))


if __name__ == "__main__":
    unittest.main()
