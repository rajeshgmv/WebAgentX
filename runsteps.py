import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field
from selenium.common.exceptions import StaleElementReferenceException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait

if __package__:
    from .identifysteps import Step, StepPlan
    from .llm import initialize_groq_llm, invoke_structured_output
else:
    from identifysteps import Step, StepPlan
    from llm import initialize_groq_llm, invoke_structured_output

logger = logging.getLogger(__name__)

USEFUL_ELEMENT_SELECTOR = ", ".join(
    (
        "a[href]",
        "button",
        "input:not([type='hidden'])",
        "select",
        "textarea",
        "[contenteditable='true']",
        "[role='button']",
        "[role='link']",
        "[role='dialog']",
        "[aria-modal='true']",
        "dialog",
    )
)

POPUP_SELECTOR = ", ".join(
    (
        "dialog",
        "[role='dialog']",
        "[aria-modal='true']",
        "[class*='modal']",
        "[class*='popup']",
        "[id*='modal']",
        "[id*='popup']",
        "[class*='consent']",
        "[id*='consent']",
        "[class*='cookie']",
        "[id*='cookie']",
    )
)


class PageElement(BaseModel):
    """A compact, safe description of a visible browser element."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=1)
    tag: str
    text: str
    type: str | None
    name: str | None
    id: str | None
    placeholder: str | None
    aria_label: str | None
    accessible_name: str | None
    role: str | None
    href: str | None
    enabled: bool
    in_popup: bool


class PageObservation(BaseModel):
    """The limited page state sent to the decision model."""

    model_config = ConfigDict(extra="forbid")

    current_url: str
    title: str
    elements: list[PageElement]


class CandidateEvaluation(BaseModel):
    """A plausible element candidate compared against the complete objective."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    index: int = Field(..., ge=1)
    reason: str = Field(..., min_length=1)


class NextStepConfirmation(BaseModel):
    """Groq's structured review of the next planned browser step."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    decision: Literal["proceed", "handle_popup", "replan", "complete"]
    planned_step_number: int | None = Field(..., ge=1)
    confirmed_step_description: str | None
    action_type: Literal["click", "none"]
    target_element_index: int | None = Field(..., ge=1)
    candidate_elements: list[CandidateEvaluation] = Field(..., max_length=5)
    blocking_element_indices: list[int]
    expected_result: str | None = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)


@dataclass(slots=True)
class PageSnapshot:
    """Serializable page information plus the local Selenium element mapping."""

    observation: PageObservation
    elements_by_index: dict[int, WebElement]


@dataclass(slots=True)
class FirstStepRunResult:
    """Result of navigation, observation, and next-step confirmation."""

    completed_step: Step
    snapshot: PageSnapshot
    confirmation: NextStepConfirmation


@dataclass(slots=True)
class StepRunResult:
    """Result of one approved plan-step action and the following review."""

    planned_step: Step
    confirmation: NextStepConfirmation
    selected_element: PageElement
    pre_action_snapshot: PageSnapshot
    post_action_snapshot: PageSnapshot
    page_changed: bool
    remapped_after_stale: bool
    next_confirmation: NextStepConfirmation | None


# Retain the old public name for callers that imported it before execution was
# generalized beyond step 2.
SecondStepRunResult = StepRunResult


def _compact_text(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _optional_text(value: object, limit: int = 200) -> str | None:
    text = _compact_text(value, limit)
    return text or None


def _safe_url(value: object, limit: int = 2_000) -> str:
    """Remove query values and fragments that may contain sensitive data."""
    url = _compact_text(value, limit)
    try:
        parsed = urlsplit(url)
    except ValueError:
        return url

    if parsed.scheme not in {"http", "https"}:
        return url
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def wait_for_page_ready(driver: WebDriver, timeout: float = 20.0) -> None:
    """Wait until navigation has produced a usable document body."""
    wait = WebDriverWait(driver, timeout)
    wait.until(
        lambda current_driver: current_driver.execute_script(
            "return document.readyState"
        )
        in {"interactive", "complete"}
    )
    wait.until(lambda current_driver: current_driver.find_elements(By.TAG_NAME, "body"))


def _is_in_popup(driver: WebDriver, element: WebElement) -> bool:
    try:
        return bool(
            driver.execute_script(
                "return Boolean(arguments[0].closest(arguments[1]));",
                element,
                POPUP_SELECTOR,
            )
        )
    except (StaleElementReferenceException, WebDriverException):
        return False


def collect_page_elements(
    driver: WebDriver,
    max_elements: int = 80,
) -> PageSnapshot:
    """Capture visible, useful elements without collecting field values or secrets."""
    if max_elements < 1:
        raise ValueError("max_elements must be at least 1")

    page_elements: list[PageElement] = []
    elements_by_index: dict[int, WebElement] = {}
    candidates = driver.find_elements(By.CSS_SELECTOR, USEFUL_ELEMENT_SELECTOR)

    for element in candidates:
        if len(page_elements) >= max_elements:
            break

        try:
            if not element.is_displayed():
                continue

            tag = _compact_text(element.tag_name, 40).lower()
            text = _compact_text(element.text, 300)
            element_type = _optional_text(element.get_attribute("type"), 80)
            name = _optional_text(element.get_attribute("name"))
            html_id = _optional_text(element.get_attribute("id"))
            placeholder = _optional_text(element.get_attribute("placeholder"))
            aria_label = _optional_text(element.get_attribute("aria-label"))
            role = _optional_text(element.get_attribute("role"), 80)
            raw_href = _optional_text(element.get_attribute("href"), 300)
            href = _safe_url(raw_href, 300) if raw_href else None
            accessible_name = _optional_text(element.accessible_name)

            # Ignore empty wrappers that provide no useful choice to the model.
            if not any(
                (
                    text,
                    element_type,
                    name,
                    html_id,
                    placeholder,
                    aria_label,
                    accessible_name,
                    role,
                    href,
                )
            ) and tag not in {"dialog", "select", "textarea"}:
                continue

            index = len(page_elements) + 1
            page_elements.append(
                PageElement(
                    index=index,
                    tag=tag,
                    text=text,
                    type=element_type,
                    name=name,
                    id=html_id,
                    placeholder=placeholder,
                    aria_label=aria_label,
                    accessible_name=accessible_name,
                    role=role,
                    href=href,
                    enabled=element.is_enabled(),
                    in_popup=_is_in_popup(driver, element),
                )
            )
            elements_by_index[index] = element
        except (StaleElementReferenceException, WebDriverException):
            logger.debug("Skipped an element that changed during page inspection")

    observation = PageObservation(
        current_url=_safe_url(driver.current_url),
        title=_compact_text(driver.title, 300),
        elements=page_elements,
    )
    return PageSnapshot(
        observation=observation,
        elements_by_index=elements_by_index,
    )


def confirm_next_step(
    plan: StepPlan,
    next_step: Step,
    snapshot: PageSnapshot,
    api_key: str | None = None,
) -> NextStepConfirmation:
    """Ask Groq whether the observed page supports the next planned step."""
    prompt_path = Path(__file__).parent / "prompts/confirmnextstep.md"
    system_prompt = prompt_path.read_text(encoding="utf-8")
    payload = {
        "normalized_objective": plan.normalized_request,
        "known_ambiguities": plan.ambiguities,
        "success_criteria": plan.success_criteria,
        "planned_next_step": next_step.model_dump(),
        "page_observation": snapshot.observation.model_dump(exclude_none=True),
    }

    llm = initialize_groq_llm(api_key=api_key, temperature=0.0)
    confirmation = invoke_structured_output(
        llm,
        NextStepConfirmation,
        [
            ("system", system_prompt),
            ("human", json.dumps(payload, ensure_ascii=False)),
        ],
    )

    if confirmation.planned_step_number != next_step.step_number:
        raise ValueError("The confirmation returned the wrong planned step number")

    candidate_indices = [candidate.index for candidate in confirmation.candidate_elements]
    if len(candidate_indices) != len(set(candidate_indices)):
        raise ValueError("The confirmation returned duplicate candidate indexes")

    valid_indices = set(snapshot.elements_by_index)
    reported_indices = set(confirmation.blocking_element_indices) | set(
        candidate_indices
    )
    if confirmation.target_element_index is not None:
        reported_indices.add(confirmation.target_element_index)
    if not reported_indices.issubset(valid_indices):
        raise ValueError("The confirmation referenced an unknown element index")

    if (
        confirmation.target_element_index is not None
        and confirmation.target_element_index not in candidate_indices
    ):
        raise ValueError("The selected target was not included in the candidate list")

    if (
        confirmation.decision in {"proceed", "handle_popup"}
        and not confirmation.expected_result
    ):
        raise ValueError("An actionable confirmation must describe its expected result")

    if confirmation.action_type == "click" and confirmation.target_element_index is None:
        raise ValueError("A click action must include a target element index")
    if confirmation.action_type == "none" and confirmation.target_element_index is not None:
        raise ValueError("A non-actionable confirmation cannot include a target element")

    return confirmation


def _page_element_by_index(snapshot: PageSnapshot, index: int) -> PageElement:
    for element in snapshot.observation.elements:
        if element.index == index:
            return element
    raise ValueError(f"Element index {index} was not found in the page observation")


def _validate_click_target(
    confirmation: NextStepConfirmation,
    snapshot: PageSnapshot,
) -> tuple[PageElement, WebElement]:
    if confirmation.decision not in {"proceed", "handle_popup"}:
        raise ValueError(
            f"Decision '{confirmation.decision}' does not authorize browser execution"
        )
    if confirmation.action_type != "click":
        raise ValueError("The proposed next action is not an executable click")
    if confirmation.target_element_index is None:
        raise ValueError("The click action does not contain a target element")

    index = confirmation.target_element_index
    page_element = _page_element_by_index(snapshot, index)
    web_element = snapshot.elements_by_index.get(index)
    if web_element is None:
        raise ValueError("The selected element is not part of the latest snapshot")

    click_input_types = {"button", "checkbox", "radio", "reset", "submit"}
    is_clickable_kind = (
        page_element.tag in {"a", "button", "summary"}
        or page_element.role in {"button", "link"}
        or (
            page_element.tag == "input"
            and page_element.type in click_input_types
        )
    )
    if not is_clickable_kind:
        raise ValueError(
            "The selected element is not represented as a clickable control"
        )

    return page_element, web_element


def _element_fingerprint(element: PageElement) -> tuple[object, ...]:
    """Build a technical identity for remapping an element after a DOM rerender."""
    return (
        element.tag,
        element.accessible_name,
        element.aria_label,
        element.text,
        element.role,
        element.href,
        element.type,
        element.placeholder,
    )


def _remap_target(
    previous_element: PageElement,
    fresh_snapshot: PageSnapshot,
) -> tuple[PageElement, WebElement] | None:
    fingerprint = _element_fingerprint(previous_element)
    matches = [
        element
        for element in fresh_snapshot.observation.elements
        if _element_fingerprint(element) == fingerprint
    ]
    if len(matches) != 1:
        return None

    remapped = matches[0]
    web_element = fresh_snapshot.elements_by_index.get(remapped.index)
    if web_element is None:
        return None
    return remapped, web_element


def _click_element(
    driver: WebDriver,
    element: WebElement,
    timeout: float,
) -> None:
    """Perform one native Selenium click without JavaScript-click fallback."""
    if not element.is_displayed():
        raise ValueError("The selected element is no longer visible")
    if not element.is_enabled():
        raise ValueError("The selected element is disabled")

    driver.execute_script(
        "arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});",
        element,
    )
    clickable = WebDriverWait(driver, timeout).until(
        lambda _driver: element
        if element.is_displayed() and element.is_enabled()
        else False
    )
    clickable.click()


def _observation_signature(observation: PageObservation) -> tuple[object, ...]:
    elements = tuple(
        _element_fingerprint(element) for element in observation.elements
    )
    return observation.current_url, observation.title, elements


def _plan_step_index(plan: StepPlan, planned_step: Step) -> int:
    """Resolve a step against the validated plan without trusting list offsets."""
    for index, step in enumerate(plan.steps):
        if step.step_number == planned_step.step_number:
            if step != planned_step:
                raise ValueError("The supplied step does not match the generated plan")
            return index
    raise ValueError("The supplied step is not part of the generated plan")


def run_next_step(
    driver: WebDriver,
    plan: StepPlan,
    planned_step: Step,
    pre_action_snapshot: PageSnapshot,
    confirmation: NextStepConfirmation,
    api_key: str | None = None,
    *,
    click_timeout: float = 10.0,
    page_load_timeout: float = 20.0,
    settle_seconds: float = 0.5,
    max_elements: int = 80,
) -> StepRunResult:
    """Execute one confirmed click and prepare the following plan step."""
    step_index = _plan_step_index(plan, planned_step)
    if step_index == 0:
        raise ValueError("The navigation step must be executed by run_first_step")
    if step_index == len(plan.steps) - 1:
        raise ValueError("The final result step is reserved for outcome handling")
    if confirmation.planned_step_number != planned_step.step_number:
        raise ValueError(
            "The proposed action does not belong to the supplied plan step"
        )

    selected_element, web_element = _validate_click_target(
        confirmation,
        pre_action_snapshot,
    )
    remapped_after_stale = False

    logger.info(
        "Starting step %s action=%s target_index=%s",
        planned_step.step_number,
        confirmation.action_type,
        selected_element.index,
    )

    previous_window_handles = set(driver.window_handles)
    try:
        _click_element(driver, web_element, timeout=click_timeout)
    except StaleElementReferenceException:
        logger.info("Selected element became stale; refreshing the page snapshot")
        fresh_snapshot = collect_page_elements(driver, max_elements=max_elements)
        remapped = _remap_target(selected_element, fresh_snapshot)

        if remapped is None:
            confirmation = confirm_next_step(
                plan,
                planned_step,
                fresh_snapshot,
                api_key=api_key,
            )
            selected_element, web_element = _validate_click_target(
                confirmation,
                fresh_snapshot,
            )
        else:
            selected_element, web_element = remapped

        pre_action_snapshot = fresh_snapshot
        remapped_after_stale = True
        _click_element(driver, web_element, timeout=click_timeout)

    current_window_handles = set(driver.window_handles)
    new_window_handles = current_window_handles - previous_window_handles
    if new_window_handles:
        driver.switch_to.window(new_window_handles.pop())

    wait_for_page_ready(driver, timeout=page_load_timeout)
    if settle_seconds > 0:
        time.sleep(settle_seconds)

    post_action_snapshot = collect_page_elements(driver, max_elements=max_elements)
    page_changed = _observation_signature(
        pre_action_snapshot.observation
    ) != _observation_signature(post_action_snapshot.observation)
    review_step = (
        planned_step
        if confirmation.decision == "handle_popup"
        else plan.steps[step_index + 1]
    )
    try:
        next_confirmation = confirm_next_step(
            plan,
            review_step,
            post_action_snapshot,
            api_key=api_key,
        )
    except Exception:
        # The browser action has already happened. Preserve its fresh snapshot
        # so the UI can retry only the LLM review instead of clicking twice.
        logger.exception(
            "Could not prepare confirmation for step %s",
            review_step.step_number,
        )
        next_confirmation = None

    logger.info(
        "Executed step %s action=%s target_index=%s page_changed=%s",
        planned_step.step_number,
        confirmation.action_type,
        selected_element.index,
        page_changed,
    )
    return StepRunResult(
        planned_step=planned_step,
        confirmation=confirmation,
        selected_element=selected_element,
        pre_action_snapshot=pre_action_snapshot,
        post_action_snapshot=post_action_snapshot,
        page_changed=page_changed,
        remapped_after_stale=remapped_after_stale,
        next_confirmation=next_confirmation,
    )


def run_second_step(
    driver: WebDriver,
    plan: StepPlan,
    first_step_result: FirstStepRunResult,
    api_key: str | None = None,
    **kwargs: object,
) -> StepRunResult:
    """Compatibility wrapper for the original fixed step-2 entry point."""
    if len(plan.steps) < 2:
        raise ValueError("The plan does not contain a second step")
    return run_next_step(
        driver=driver,
        plan=plan,
        planned_step=plan.steps[1],
        pre_action_snapshot=first_step_result.snapshot,
        confirmation=first_step_result.confirmation,
        api_key=api_key,
        **kwargs,
    )


def run_first_step(
    driver: WebDriver,
    plan: StepPlan,
    api_key: str | None = None,
    *,
    page_load_timeout: float = 20.0,
    settle_seconds: float = 0.5,
    max_elements: int = 80,
) -> FirstStepRunResult:
    """Navigate to the plan URL, inspect the page, and review the next step."""
    first_step = plan.steps[0]
    logger.info(
        "Starting step %s: navigate to %s",
        first_step.step_number,
        plan.url,
    )

    driver.get(plan.url)
    wait_for_page_ready(driver, timeout=page_load_timeout)
    if settle_seconds > 0:
        time.sleep(settle_seconds)

    snapshot = collect_page_elements(driver, max_elements=max_elements)
    logger.info(
        "Completed step %s: current_url=%s observed_elements=%s",
        first_step.step_number,
        snapshot.observation.current_url,
        len(snapshot.observation.elements),
    )

    if len(plan.steps) == 1:
        confirmation = NextStepConfirmation(
            decision="complete",
            planned_step_number=first_step.step_number,
            confirmed_step_description=first_step.description,
            action_type="none",
            target_element_index=None,
            candidate_elements=[],
            blocking_element_indices=[],
            expected_result=None,
            reason="The generated plan contains no additional steps.",
        )
    else:
        confirmation = confirm_next_step(
            plan,
            plan.steps[1],
            snapshot,
            api_key=api_key,
        )

    logger.info(
        "Next-step review: decision=%s planned_step=%s target_index=%s",
        confirmation.decision,
        confirmation.planned_step_number,
        confirmation.target_element_index,
    )
    return FirstStepRunResult(
        completed_step=first_step,
        snapshot=snapshot,
        confirmation=confirmation,
    )
