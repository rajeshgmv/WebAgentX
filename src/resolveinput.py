"""Field-aware validation and normalization for runtime user input."""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if __package__:
    from .llm import initialize_groq_llm, invoke_structured_output
else:
    from llm import initialize_groq_llm, invoke_structured_output


class GeneratedInputResolution(BaseModel):
    """The transformation fields generated through Groq function calling."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    resolved_value: str = Field(..., min_length=1, max_length=500)
    interpretation: str = Field(..., min_length=1, max_length=500)
    confidence: Literal["high", "medium", "low"]
    ambiguity: str | None = Field(default=None, min_length=1, max_length=300)


class InputResolution(GeneratedInputResolution):
    """A generated resolution tied to the exact value supplied by the user."""

    original_value: str = Field(..., min_length=1, max_length=500)
    changed: bool


def resolve_required_input(
    raw_value: str,
    context: dict[str, Any],
    api_key: str | None = None,
) -> InputResolution:
    """Interpret newly supplied input using its current field and task context."""
    original_value = raw_value.strip()
    if not original_value:
        raise ValueError("Required input cannot be empty")
    if len(original_value) > 500:
        raise ValueError("Required input must be 500 characters or fewer")

    prompt_path = Path(__file__).parent / "prompts/resolveinput.md"
    system_prompt = prompt_path.read_text(encoding="utf-8")
    llm = initialize_groq_llm(
        api_key=api_key,
        temperature=0.0,
        max_tokens=600,
    )
    generated = invoke_structured_output(
        llm,
        GeneratedInputResolution,
        [
            ("system", system_prompt),
            (
                "human",
                json.dumps(
                    {
                        "user_supplied_value": original_value,
                        "required_input_context": context,
                    },
                    ensure_ascii=False,
                ),
            ),
        ],
    )
    resolved_value = generated.resolved_value.strip()
    return InputResolution(
        original_value=original_value,
        resolved_value=resolved_value,
        changed=resolved_value != original_value,
        interpretation=generated.interpretation,
        confidence=generated.confidence,
        ambiguity=generated.ambiguity,
    )
