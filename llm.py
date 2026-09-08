import logging
import os
import re
import time
from typing import TypeVar

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel

logger = logging.getLogger(__name__)

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)

# Support command-line use as well as the Streamlit application.
load_dotenv()


def initialize_groq_llm(
    api_key: str | None = None,
    model: str = "openai/gpt-oss-20b",
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> ChatGroq:
    """Initialize a Groq chat model without modifying process environment."""
    resolved_api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY")

    if not resolved_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=resolved_api_key,
        max_tokens=max_tokens,
        max_retries=2,
        reasoning_format="hidden",
        reasoning_effort="low",
    )


def _short_rate_limit_delay(error: Exception) -> float | None:
    """Return a safe retry delay for a brief Groq 429, if advertised."""
    if getattr(error, "status_code", None) != 429:
        return None

    match = re.search(
        r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|s)\b",
        str(error),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    delay = float(match.group(1))
    if match.group(2).lower() == "ms":
        delay /= 1_000

    # Longer limits should return control to the UI instead of blocking it.
    if delay > 10.0:
        return None
    return max(delay, 0.05)


def _invoke_with_short_rate_limit_retry(
    runnable: object,
    messages: list[tuple[str, str]],
) -> object:
    """Retry once only when Groq explicitly requests a brief wait."""
    try:
        return runnable.invoke(messages)  # type: ignore[attr-defined]
    except Exception as error:
        delay = _short_rate_limit_delay(error)
        if delay is None:
            raise
        logger.warning(
            "Groq temporarily rate-limited the request; retrying in %.3f seconds",
            delay,
        )
        time.sleep(delay)
        return runnable.invoke(messages)  # type: ignore[attr-defined]


def _is_repairable_structured_output_error(error: Exception) -> bool:
    """Recognize provider-side schema rejection before LangChain can parse it."""
    if getattr(error, "status_code", None) != 400:
        return False
    normalized = str(error).casefold()
    return any(
        marker in normalized
        for marker in (
            "tool_use_failed",
            "tool call validation failed",
            "json_validate_failed",
            "did not match schema",
        )
    )


def _schema_repair_instruction(
    schema: type[StructuredModel],
    error: Exception,
) -> str:
    summary = " ".join(str(error).split())[:1_200]
    return (
        f"Your previous {schema.__name__} function call was rejected by the "
        "provider because its arguments did not match the schema. Call the "
        "function again with schema-valid arguments. Never invent sentinel "
        "values such as element index 0; use an empty candidate list, null "
        "target, action_type none, and an appropriate non-action decision when "
        "no observed element exists. Provider error: "
        f"{summary}"
    )


def _tool_call_value(tool_call: object, key: str) -> object:
    if isinstance(tool_call, dict):
        return tool_call.get(key)
    return getattr(tool_call, key, None)


def _recover_namespaced_tool_call(
    raw_message: object,
    schema: type[StructuredModel],
) -> StructuredModel | None:
    """Validate a Groq call that only differs by its `functions.` namespace."""
    tool_calls = getattr(raw_message, "tool_calls", None) or []
    schema_name = schema.model_config.get("title") or schema.__name__
    accepted_names = {schema_name, f"functions.{schema_name}"}
    matching_calls = [
        tool_call
        for tool_call in tool_calls
        if _tool_call_value(tool_call, "name") in accepted_names
    ]
    if len(tool_calls) != 1 or len(matching_calls) != 1:
        return None

    arguments = _tool_call_value(matching_calls[0], "args")
    if not isinstance(arguments, dict):
        return None
    if _tool_call_value(matching_calls[0], "name") != schema_name:
        logger.warning(
            "Groq namespaced the structured function as functions.%s; "
            "validating its arguments locally",
            schema_name,
        )
    return schema.model_validate(arguments)


def _validate_structured_response(
    response: object,
    schema: type[StructuredModel],
) -> StructuredModel | None:
    """Validate normal responses and recover Groq's namespaced tool-name variant."""
    if response is None:
        return None

    if isinstance(response, dict) and {
        "raw",
        "parsed",
        "parsing_error",
    }.issubset(response):
        parsed = response.get("parsed")
        if parsed is not None:
            return schema.model_validate(parsed)

        recovered = _recover_namespaced_tool_call(response.get("raw"), schema)
        if recovered is not None:
            return recovered

        parsing_error = response.get("parsing_error")
        if isinstance(parsing_error, BaseException):
            raise parsing_error
        return None

    return schema.model_validate(response)


def invoke_structured_output(
    llm: ChatGroq,
    schema: type[StructuredModel],
    messages: list[tuple[str, str]],
) -> StructuredModel:
    """Invoke Groq function calling and validate the returned arguments locally."""
    structured_llm = llm.with_structured_output(
        schema,
        method="function_calling",
        include_raw=True,
    )
    repaired_provider_error = False
    try:
        response = _invoke_with_short_rate_limit_retry(structured_llm, messages)
    except Exception as error:
        if not _is_repairable_structured_output_error(error):
            raise
        repaired_provider_error = True
        logger.warning(
            "Groq rejected schema-invalid tool arguments; retrying once"
        )
        response = _invoke_with_short_rate_limit_retry(
            structured_llm,
            [*messages, ("human", _schema_repair_instruction(schema, error))],
        )
    validated = _validate_structured_response(response, schema)
    if validated is None:
        if repaired_provider_error:
            raise ValueError(
                "Groq omitted the structured function call after schema repair"
            )
        logger.warning(
            "Groq omitted the required structured function call; retrying once"
        )
        repair_instruction = (
            f"Your previous response omitted the required {schema.__name__} "
            "function call. Call that function now, supply every required "
            "argument, and do not answer with plain text."
        )
        response = _invoke_with_short_rate_limit_retry(
            structured_llm,
            [*messages, ("human", repair_instruction)],
        )
        validated = _validate_structured_response(response, schema)
    if validated is None:
        raise ValueError(
            "Groq did not return the required structured function call after retry"
        )
    return validated
