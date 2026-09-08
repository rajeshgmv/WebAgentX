import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict

from src.llm import invoke_structured_output


class ExampleOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[dict[str, str]]


class FakeRunnable:
    def __init__(self, *, response=None, error=None):
        self.response = response
        self.error = error
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.response


class FunctionCallingLlm:
    def __init__(self):
        self.structured = FakeRunnable(response={"items": [{"name": "value"}]})
        self.calls = []

    def with_structured_output(self, schema, **options):
        self.calls.append((schema, options))
        return self.structured


class FailingLlm:
    def with_structured_output(self, schema, **options):
        return FakeRunnable(error=RuntimeError("authentication failed"))


class ShortRateLimitError(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("Please try again in 225ms")


class RateLimitedRunnable:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            raise ShortRateLimitError()
        return {"items": [{"name": "value"}]}


class RateLimitedLlm:
    def __init__(self):
        self.structured = RateLimitedRunnable()

    def with_structured_output(self, schema, **options):
        return self.structured


class MissingThenValidRunnable:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return None
        return {"items": [{"name": "value"}]}


class MissingThenValidLlm:
    def __init__(self):
        self.structured = MissingThenValidRunnable()

    def with_structured_output(self, schema, **options):
        return self.structured


class NamespacedToolLlm:
    def with_structured_output(self, schema, **options):
        raw = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "functions.ExampleOutput",
                    "args": {"items": [{"name": "value"}]},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
        return FakeRunnable(
            response={
                "raw": raw,
                "parsed": None,
                "parsing_error": RuntimeError("unknown tool type"),
            }
        )


class SchemaRejectedError(Exception):
    status_code = 400

    def __init__(self):
        super().__init__(
            "tool_use_failed: tool call validation failed: index must be >= 1"
        )


class SchemaRejectedThenValidRunnable:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            raise SchemaRejectedError()
        return {"items": [{"name": "value"}]}


class SchemaRejectedThenValidLlm:
    def __init__(self):
        self.structured = SchemaRejectedThenValidRunnable()

    def with_structured_output(self, schema, **options):
        return self.structured


class StructuredOutputTests(unittest.TestCase):
    def test_retries_provider_rejected_schema_arguments_once(self):
        llm = SchemaRejectedThenValidLlm()

        result = invoke_structured_output(
            llm,
            ExampleOutput,
            [("system", "Return structured data")],
        )

        self.assertEqual(result.items, [{"name": "value"}])
        self.assertEqual(len(llm.structured.calls), 2)
        self.assertIn(
            "Never invent sentinel values such as element index 0",
            llm.structured.calls[1][-1][1],
        )

    def test_uses_function_calling_and_validates_the_result(self):
        llm = FunctionCallingLlm()

        result = invoke_structured_output(
            llm,
            ExampleOutput,
            [("system", "Return structured data")],
        )

        self.assertEqual(result.items, [{"name": "value"}])
        self.assertEqual(
            [options for _, options in llm.calls],
            [{"method": "function_calling", "include_raw": True}],
        )

    def test_accepts_only_the_expected_functions_namespace_prefix(self):
        result = invoke_structured_output(
            NamespacedToolLlm(),
            ExampleOutput,
            [("system", "Return structured data")],
        )

        self.assertEqual(result.items, [{"name": "value"}])

    def test_does_not_retry_unrelated_errors(self):
        with self.assertRaisesRegex(RuntimeError, "authentication failed"):
            invoke_structured_output(
                FailingLlm(),
                ExampleOutput,
                [("system", "Return structured data")],
            )

    def test_retries_once_after_short_advertised_rate_limit(self):
        llm = RateLimitedLlm()

        with patch("src.llm.time.sleep") as sleep:
            result = invoke_structured_output(
                llm,
                ExampleOutput,
                [("system", "Return structured data")],
            )

        self.assertEqual(result.items, [{"name": "value"}])
        self.assertEqual(llm.structured.calls, 2)
        sleep.assert_called_once_with(0.225)

    def test_retries_when_model_omits_the_required_function_call(self):
        llm = MissingThenValidLlm()

        result = invoke_structured_output(
            llm,
            ExampleOutput,
            [("system", "Return structured data")],
        )

        self.assertEqual(result.items, [{"name": "value"}])
        self.assertEqual(len(llm.structured.calls), 2)
        self.assertIn(
            "do not answer with plain text",
            llm.structured.calls[1][-1][1],
        )


if __name__ == "__main__":
    unittest.main()
