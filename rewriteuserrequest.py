import json
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

if __package__:
    from .llm import initialize_groq_llm, invoke_structured_output
else:
    from llm import initialize_groq_llm, invoke_structured_output

logger = logging.getLogger(__name__)


class ProvidedInput(BaseModel):
    """One explicit user-supplied value that may map to a page control."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    label: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="A domain-neutral semantic label for the supplied value",
    )
    value: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The exact value as written in the original request",
    )


class RewrittenUserRequest(BaseModel):
    """A goal-oriented interpretation of the original browser request."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    normalized_request: str = Field(
        ...,
        min_length=1,
        description="The user's intended outcome without assumed UI controls",
    )
    ambiguities: list[str] = Field(
        ...,
        max_length=10,
        description="Uncertainties that may affect execution, or an empty list",
    )
    provided_inputs: list[ProvidedInput] = Field(
        ...,
        max_length=20,
        description="Distinct explicit values that may need separate controls",
    )


def rewrite_user_request(
    user_request: str,
    api_key: str | None = None,
) -> RewrittenUserRequest:
    """Rewrite a request as an outcome without selecting a browser mechanism."""
    user_request = user_request.strip()
    if not user_request:
        raise ValueError("User request cannot be empty")

    prompt_path = Path(__file__).parent / "prompts/rewriteuserrequest.md"
    system_prompt = prompt_path.read_text(encoding="utf-8")

    llm = initialize_groq_llm(
        api_key=api_key,
        temperature=0.0,
        max_tokens=768,
    )
    rewritten = invoke_structured_output(
        llm,
        RewrittenUserRequest,
        [
            ("system", system_prompt),
            (
                "human",
                json.dumps({"original_user_request": user_request}, ensure_ascii=False),
            ),
        ],
    )

    normalized_source = " ".join(user_request.casefold().split())
    verified_inputs: list[ProvidedInput] = []
    for provided_input in rewritten.provided_inputs:
        normalized_value = " ".join(provided_input.value.casefold().split())
        if normalized_value and normalized_value in normalized_source:
            verified_inputs.append(provided_input)
        else:
            logger.warning(
                "Discarded an input value not grounded in the original request: %s",
                provided_input.label,
            )

    return rewritten.model_copy(update={"provided_inputs": verified_inputs})
