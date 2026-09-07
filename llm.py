import os
import logging
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
) -> ChatGroq:
    """Initialize a Groq chat model without modifying process environment."""
    resolved_api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY")

    if not resolved_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=resolved_api_key,
        max_retries=2,
        reasoning_format="hidden",
    )


def _is_json_schema_generation_error(error: Exception) -> bool:
    """Identify Groq's schema-generation failure without retrying other 400s."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        details = body.get("error", body)
        if isinstance(details, dict) and details.get("code") == "json_validate_failed":
            return True
    return "json_validate_failed" in str(error)


def invoke_structured_output(
    llm: ChatGroq,
    schema: type[StructuredModel],
    messages: list[tuple[str, str]],
) -> StructuredModel:
    """Invoke strict Groq output with a targeted function-calling fallback."""
    strict_llm = llm.with_structured_output(
        schema,
        method="json_schema",
        strict=True,
    )

    try:
        response = strict_llm.invoke(messages)
    except Exception as error:
        if not _is_json_schema_generation_error(error):
            raise

        logger.warning(
            "Groq generated schema-invalid JSON; retrying with function calling"
        )
        repair_instruction = (
            "Regenerate the response using the required schema. Every array item "
            "declared as an object must be an actual JSON object, never a quoted or "
            "escaped JSON string. Preserve the requested meaning."
        )
        fallback_llm = llm.with_structured_output(
            schema,
            method="function_calling",
        )
        response = fallback_llm.invoke([*messages, ("human", repair_instruction)])

    return schema.model_validate(response)
