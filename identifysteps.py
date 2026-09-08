from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

if __package__:
    from .llm import initialize_groq_llm, invoke_structured_output
    from .rewriteuserrequest import ProvidedInput, rewrite_user_request
else:
    from llm import initialize_groq_llm, invoke_structured_output
    from rewriteuserrequest import ProvidedInput, rewrite_user_request


class Step(BaseModel):
    """A single high-level step in the browser task."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    step_number: int = Field(
        ...,
        ge=1,
        description="The one-based step number in the sequence",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="A concise, high-level browser action",
    )


class GeneratedStepPlan(BaseModel):
    """The plan fields generated and validated through Groq."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    steps: list[Step] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="An ordered, adaptable browser plan",
    )
    success_criteria: str = Field(
        ...,
        min_length=1,
        description="The observable condition that means the task is complete",
    )

    @model_validator(mode="after")
    def validate_step_numbers(self) -> "GeneratedStepPlan":
        expected = list(range(1, len(self.steps) + 1))
        actual = [step.step_number for step in self.steps]
        if actual != expected:
            raise ValueError("Step numbers must be sequential and start at 1")
        return self


class StepPlan(BaseModel):
    """A generated plan combined with the original user input."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    url: str
    user_request: str = Field(..., min_length=1)
    normalized_request: str = Field(..., min_length=1)
    ambiguities: list[str] = Field(..., max_length=10)
    provided_inputs: list[ProvidedInput] = Field(default_factory=list, max_length=20)
    steps: list[Step] = Field(..., min_length=1, max_length=20)
    success_criteria: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> "StepPlan":
        if not is_valid_http_url(self.url):
            raise ValueError("URL must be a valid HTTP or HTTPS URL")

        expected = list(range(1, len(self.steps) + 1))
        actual = [step.step_number for step in self.steps]
        if actual != expected:
            raise ValueError("Step numbers must be sequential and start at 1")
        return self


def is_valid_http_url(url: str) -> bool:
    """Return whether a string is a complete HTTP(S) URL."""
    if not url or any(character.isspace() for character in url):
        return False

    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname)
    except ValueError:
        return False


def identify_steps(
    url: str,
    user_request: str,
    api_key: str | None = None,
) -> StepPlan:
    """Generate a validated browser task plan using Groq structured output."""
    url = url.strip()
    user_request = user_request.strip()

    if not is_valid_http_url(url):
        raise ValueError("URL must be a valid HTTP or HTTPS URL")
    if not user_request:
        raise ValueError("User request cannot be empty")

    rewritten_request = rewrite_user_request(user_request, api_key=api_key)

    prompt_path = Path(__file__).parent / "prompts/generatesteps.md"
    system_prompt = prompt_path.read_text(encoding="utf-8")

    llm = initialize_groq_llm(
        api_key=api_key,
        temperature=0.0,
        max_tokens=1_200,
    )
    generated_plan = invoke_structured_output(
        llm,
        GeneratedStepPlan,
        [
            ("system", system_prompt),
            (
                "human",
                (
                    f"Website URL:\n{url}\n\n"
                    "Normalized browser objective:\n"
                    f"{rewritten_request.normalized_request}\n\n"
                    "Known ambiguities:\n"
                    f"{rewritten_request.ambiguities}\n\n"
                    "Distinct user-provided inputs:\n"
                    f"{[item.model_dump() for item in rewritten_request.provided_inputs]}"
                ),
            ),
        ],
    )

    # Preserve submitted metadata instead of trusting the model to reproduce it.
    return StepPlan(
        url=url,
        user_request=user_request,
        normalized_request=rewritten_request.normalized_request,
        ambiguities=rewritten_request.ambiguities,
        provided_inputs=rewritten_request.provided_inputs,
        steps=generated_plan.steps,
        success_criteria=generated_plan.success_criteria,
    )


def main() -> None:
    plan = identify_steps(
        url="https://www.aurorahealthcare.org",
        user_request=(
            "Search for a doctor named Swapna and return the doctor's details."
        ),
    )
    print("\nGenerated Steps:\n")
    print(plan)


if __name__ == "__main__":
    main()
