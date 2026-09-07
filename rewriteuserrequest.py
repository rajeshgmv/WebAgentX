import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

if __package__:
    from .llm import initialize_groq_llm, invoke_structured_output
else:
    from llm import initialize_groq_llm, invoke_structured_output


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

    llm = initialize_groq_llm(api_key=api_key, temperature=0.0)
    return invoke_structured_output(
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
