import json
import logging
import time
from dataclasses import dataclass, field
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

MAX_PROMPT_ELEMENTS = 50
MAX_PROMPT_ELEMENT_CHARS = 8_000
MAX_VISIBLE_TEXT_CHARS = 10_000

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
        "[role='combobox']",
        "[role='listbox']",
        "[role='option']",
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

OPEN_SHADOW_QUERY_SCRIPT = """
/* collect-elements-across-open-shadow-roots */
const selector = arguments[0];
const results = [];
const visitedRoots = new Set();

function collect(root) {
    if (!root || visitedRoots.has(root)) {
        return;
    }
    visitedRoots.add(root);

    for (const element of root.querySelectorAll("*")) {
        if (element.matches(selector)) {
            results.push(element);
        }
        if (element.shadowRoot && element.shadowRoot.mode === "open") {
            collect(element.shadowRoot);
        }
    }
}

collect(document);
return results;
"""

OPEN_SHADOW_SIGNATURE_SCRIPT = r"""
/* describe-useful-elements-across-open-shadow-roots */
const selector = arguments[0];
const results = [];
const visitedRoots = new Set();

function compact(value, limit = 200) {
    return String(value || "").replace(/\s+/g, " ").trim().slice(0, limit);
}

function collect(root) {
    if (!root || visitedRoots.has(root)) {
        return;
    }
    visitedRoots.add(root);

    for (const element of root.querySelectorAll("*")) {
        if (element.matches(selector)) {
            const style = window.getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            results.push([
                element.tagName,
                element.getAttribute("type"),
                element.getAttribute("role"),
                element.getAttribute("placeholder"),
                element.getAttribute("aria-label"),
                element.getAttribute("aria-expanded"),
                typeof element.value === "string" ? element.value.length : null,
                compact(element.innerText || element.textContent),
                style.display,
                style.visibility,
                style.opacity,
                Math.round(rect.width),
                Math.round(rect.height),
            ]);
        }
        if (element.shadowRoot && element.shadowRoot.mode === "open") {
            collect(element.shadowRoot);
        }
    }
}

collect(document);
return JSON.stringify(results);
"""

OPEN_SHADOW_TEXT_ENTRY_SCRIPT = r"""
/* type-text-into-open-shadow-root */
const element = arguments[0];
const text = String(arguments[1]);
const root = element.getRootNode();

if (!root || !root.host || root.mode !== "open") {
    return "not-open-shadow-root";
}
if (!(element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement)) {
    return "not-text-field";
}
if (element.disabled || element.readOnly) {
    return "not-editable";
}

const prototype = element instanceof HTMLTextAreaElement
    ? HTMLTextAreaElement.prototype
    : HTMLInputElement.prototype;
const valueSetter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
if (!valueSetter) {
    return "missing-value-setter";
}

function dispatchInput(nextValue, data, inputType) {
    valueSetter.call(element, nextValue);
    let event;
    try {
        event = new InputEvent("input", {
            bubbles: true,
            composed: true,
            data,
            inputType,
        });
    } catch (_error) {
        event = new Event("input", {bubbles: true, composed: true});
    }
    element.dispatchEvent(event);
}

element.scrollIntoView({block: "center", inline: "nearest"});
element.click();
element.focus({preventScroll: true});
dispatchInput("", null, "deleteContentBackward");

let entered = "";
for (const character of text) {
    entered += character;
    dispatchInput(entered, character, "insertText");
}

return element.value === text ? "applied" : "value-mismatch";
"""

VISIBLE_PAGE_TEXT_SCRIPT = r"""
/* collect-visible-page-text */
const root = document.querySelector("main") || document.body;
return root ? root.innerText : "";
"""


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
    required: bool | None = None
    has_value: bool | None = None


class PageObservation(BaseModel):
    """The limited page state sent to the decision model."""

    model_config = ConfigDict(extra="forbid")

    current_url: str
    title: str
    elements: list[PageElement]
    visible_text: str = ""


class InputDiagnostic(BaseModel):
    """Temporary local diagnostics for one top-level input element."""

    model_config = ConfigDict(extra="forbid")

    position: int = Field(..., ge=1)
    data_testid: str | None
    type: str | None
    placeholder: str | None
    role: str | None
    matched_useful_selector: bool | None
    displayed: bool | None
    enabled: bool | None
    accessible_name: str | None
    included_index: int | None = Field(..., ge=1)
    collection_status: Literal[
        "included",
        "hidden",
        "empty",
        "error",
        "not_seen",
        "max_elements_reached",
    ]
    collection_error: str | None
    diagnostic_errors: list[str]


class CandidateEvaluation(BaseModel):
    """A plausible element candidate compared against the complete objective."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    index: int = Field(..., ge=1)
    reason: str = Field(..., min_length=1)


class NextStepConfirmation(BaseModel):
    """Groq's structured review of the next planned browser step."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    decision: Literal[
        "proceed",
        "handle_popup",
        "request_input",
        "skip_ahead",
        "replan",
        "complete",
    ]
    planned_step_number: int | None = Field(..., ge=1)
    confirmed_step_description: str | None
    action_type: Literal["click", "type_text", "none"]
    target_element_index: int | None = Field(..., ge=1)
    input_text: str | None = Field(..., min_length=1, max_length=500)
    step_progress: Literal["continue_current_step", "advance_to_next_step"]
    candidate_elements: list[CandidateEvaluation] = Field(..., max_length=5)
    blocking_element_indices: list[int]
    expected_result: str | None = Field(..., min_length=1)
    requested_input: str | None = Field(default=None, min_length=1, max_length=300)
    next_step_number: int | None = Field(default=None, ge=1)
    reason: str = Field(..., min_length=1)


class ExecutedAction(BaseModel):
    """A compact record of the last browser action known to have completed."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed"] = "completed"
    planned_step_number: int = Field(..., ge=1)
    action_type: Literal["click", "type_text"]
    target_element: PageElement
    input_text: str | None = None


class ExtractedDetail(BaseModel):
    """One fact grounded in the visible final page."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    label: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., min_length=1, max_length=1_000)


class ExtractedRecord(BaseModel):
    """One distinct item from a multi-result final page."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(..., min_length=1, max_length=300)
    details: list[ExtractedDetail] = Field(..., min_length=1, max_length=25)


class FinalTaskResult(BaseModel):
    """Structured final result extracted from the current page."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    completed: bool
    summary: str = Field(..., min_length=1, max_length=2_000)
    details: list[ExtractedDetail] = Field(..., max_length=100)
    records: list[ExtractedRecord] = Field(default_factory=list, max_length=50)
    missing_information: list[str] = Field(..., max_length=10)


@dataclass(slots=True)
class PageSnapshot:
    """Serializable page information plus the local Selenium element mapping."""

    observation: PageObservation
    elements_by_index: dict[int, WebElement]
    # Diagnostics are deliberately outside PageObservation, so they remain
    # local and are never included in the payload sent to Groq.
    input_diagnostics: list[InputDiagnostic] = field(default_factory=list)


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

    @property
    def executed_action(self) -> ExecutedAction:
        """Describe this result without exposing any newly scraped field value."""
        return ExecutedAction(
            planned_step_number=self.planned_step.step_number,
            action_type=self.confirmation.action_type,
            target_element=self.selected_element,
            input_text=self.confirmation.input_text,
        )


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


def _bounded_visible_text(value: object) -> str:
    """Preserve useful line structure while bounding final extraction input."""
    lines: list[str] = []
    previous = None
    for raw_line in str(value or "").splitlines():
        line = " ".join(raw_line.split())
        if not line or line == previous:
            continue
        lines.append(line)
        previous = line
    return "\n".join(lines)[:MAX_VISIBLE_TEXT_CHARS]


def _collect_visible_page_text(driver: WebDriver) -> str:
    try:
        return _bounded_visible_text(
            driver.execute_script(VISIBLE_PAGE_TEXT_SCRIPT)
        )
    except WebDriverException as error:
        logger.warning("Could not collect visible page text: %s", error)
        return ""


def _element_prompt_payload(element: PageElement) -> dict[str, object]:
    """Serialize an element compactly while retaining generic interaction state."""
    payload = element.model_dump(exclude_none=True)

    # True/enabled and false/not-in-popup are overwhelmingly common defaults.
    # Their absence is documented in the decision prompt.
    if payload.get("enabled") is True:
        payload.pop("enabled")
    if payload.get("in_popup") is False:
        payload.pop("in_popup")
    if payload.get("required") is False:
        payload.pop("required")

    # Accessibility APIs frequently repeat the same label in several fields.
    seen_labels: set[str] = set()
    for key in ("text", "placeholder", "aria_label", "accessible_name"):
        value = payload.get(key)
        if not isinstance(value, str):
            continue
        normalized = " ".join(value.casefold().split())
        if normalized in seen_labels:
            payload.pop(key)
        else:
            seen_labels.add(normalized)

    return payload


def _observation_prompt_payload(observation: PageObservation) -> dict[str, object]:
    """Build a generic, interaction-prioritized page payload for Groq."""
    def priority(element: PageElement) -> int:
        if element.in_popup or element.required is True:
            return 0
        if (
            element.tag in {"input", "select", "textarea", "dialog"}
            or element.role in {"combobox", "dialog", "listbox", "option"}
        ):
            return 1
        if (
            element.tag == "button"
            or element.role in {"button", "menuitem", "tab"}
        ):
            return 2
        if element.tag == "a" or element.role == "link":
            return 3
        return 4

    ranked_elements = sorted(
        observation.elements,
        key=lambda element: (priority(element), element.index),
    )
    included: list[dict[str, object]] = []
    used_chars = 0
    for element in ranked_elements:
        compact_element = _element_prompt_payload(element)
        element_chars = len(
            json.dumps(compact_element, ensure_ascii=False, separators=(",", ":"))
        )
        if len(included) >= MAX_PROMPT_ELEMENTS:
            break
        if included and used_chars + element_chars > MAX_PROMPT_ELEMENT_CHARS:
            continue
        included.append(compact_element)
        used_chars += element_chars

    # Restore DOM order after budget selection so nearby controls remain easy
    # for the model to reason about.
    included.sort(key=lambda element: int(element["index"]))
    return {
        "current_url": observation.current_url,
        "title": observation.title,
        "observed_element_count": len(observation.elements),
        "included_element_count": len(included),
        "omitted_element_count": len(observation.elements) - len(included),
        "elements": included,
    }


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


def _find_elements_across_open_shadow_roots(
    driver: WebDriver,
    selector: str,
) -> list[WebElement]:
    """Find matching elements in the document and all nested open shadow roots."""
    try:
        result = driver.execute_script(OPEN_SHADOW_QUERY_SCRIPT, selector)
        if not isinstance(result, list):
            raise WebDriverException(
                "Shadow-aware element query did not return an element list"
            )
        return result
    except WebDriverException as error:
        logger.warning(
            "Shadow-aware query failed; falling back to the document DOM: %s",
            error,
        )
        return driver.find_elements(By.CSS_SELECTOR, selector)


def _useful_element_signature(driver: WebDriver) -> str:
    """Return a value-sensitive signature of useful light and shadow DOM."""
    result = driver.execute_script(
        OPEN_SHADOW_SIGNATURE_SCRIPT,
        USEFUL_ELEMENT_SELECTOR,
    )
    if not isinstance(result, str):
        raise WebDriverException("DOM stability query did not return a signature")
    return result


def wait_for_interactive_dom_stable(
    driver: WebDriver,
    *,
    timeout: float = 6.0,
    stable_for: float = 1.0,
    poll_interval: float = 0.25,
) -> None:
    """Wait until useful light/shadow DOM remains unchanged for a short period."""
    if timeout <= 0:
        return
    if stable_for < 0:
        raise ValueError("stable_for cannot be negative")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")

    started_at = time.monotonic()
    deadline = started_at + timeout
    previous_signature: str | None = None
    stable_since = started_at

    while True:
        signature = _useful_element_signature(driver)
        now = time.monotonic()
        if signature != previous_signature:
            previous_signature = signature
            stable_since = now
        elif now - stable_since >= stable_for:
            return

        if now >= deadline:
            logger.warning(
                "Interactive DOM did not remain stable for %.2f seconds "
                "within the %.2f-second timeout",
                stable_for,
                timeout,
            )
            return

        time.sleep(min(poll_interval, max(0.0, deadline - now)))


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


def _exception_summary(stage: str, error: Exception) -> str:
    message = _compact_text(error, 240)
    return f"{stage}: {type(error).__name__}: {message or 'no message'}"


def _diagnose_input_elements(
    driver: WebDriver,
    elements_by_index: dict[int, WebElement],
    collection_statuses: list[
        tuple[WebElement, str, str | None, int | None]
    ],
    *,
    max_elements_reached: bool,
) -> list[InputDiagnostic]:
    """Inspect inputs separately to explain why collection omitted one."""
    try:
        inputs = _find_elements_across_open_shadow_roots(driver, "input")
    except (StaleElementReferenceException, WebDriverException) as error:
        logger.warning("Input diagnostic query failed: %s", error)
        return []

    useful_inputs: list[WebElement] | None
    selector_error: str | None = None
    try:
        useful_inputs = _find_elements_across_open_shadow_roots(
            driver,
            "input:not([type='hidden'])",
        )
    except (StaleElementReferenceException, WebDriverException) as error:
        useful_inputs = None
        selector_error = _exception_summary("useful selector", error)

    diagnostics: list[InputDiagnostic] = []
    for position, element in enumerate(inputs, start=1):
        diagnostic_errors: list[str] = []
        if selector_error:
            diagnostic_errors.append(selector_error)

        def read(stage: str, operation):
            try:
                return operation()
            except (StaleElementReferenceException, WebDriverException) as error:
                diagnostic_errors.append(_exception_summary(stage, error))
                return None

        matched_useful_selector = (
            None if useful_inputs is None else element in useful_inputs
        )
        displayed = read("is_displayed", element.is_displayed)
        enabled = read("is_enabled", element.is_enabled)
        element_type = _optional_text(
            read("type", lambda: element.get_attribute("type")),
            80,
        )
        placeholder = _optional_text(
            read("placeholder", lambda: element.get_attribute("placeholder"))
        )
        role = _optional_text(
            read("role", lambda: element.get_attribute("role")),
            80,
        )
        data_testid = _optional_text(
            read("data-testid", lambda: element.get_attribute("data-testid"))
        )
        accessible_name = _optional_text(
            read("accessible_name", lambda: element.accessible_name)
        )

        matching_status = next(
            (
                status
                for candidate, *status in collection_statuses
                if candidate == element
            ),
            None,
        )
        if matching_status is None:
            collection_status = (
                "max_elements_reached"
                if max_elements_reached
                else "not_seen"
            )
            collection_error = None
            included_index = next(
                (
                    index
                    for index, candidate in elements_by_index.items()
                    if candidate == element
                ),
                None,
            )
        else:
            collection_status, collection_error, included_index = matching_status

        diagnostics.append(
            InputDiagnostic(
                position=position,
                data_testid=data_testid,
                type=element_type,
                placeholder=placeholder,
                role=role,
                matched_useful_selector=matched_useful_selector,
                displayed=displayed,
                enabled=enabled,
                accessible_name=accessible_name,
                included_index=included_index,
                collection_status=collection_status,
                collection_error=collection_error,
                diagnostic_errors=diagnostic_errors,
            )
        )

    return diagnostics


def collect_page_elements(
    driver: WebDriver,
    max_elements: int = 80,
) -> PageSnapshot:
    """Capture visible, useful elements without collecting field values or secrets."""
    if max_elements < 1:
        raise ValueError("max_elements must be at least 1")

    page_elements: list[PageElement] = []
    elements_by_index: dict[int, WebElement] = {}
    input_collection_statuses: list[
        tuple[WebElement, str, str | None, int | None]
    ] = []
    max_elements_reached = False
    candidates = _find_elements_across_open_shadow_roots(
        driver,
        USEFUL_ELEMENT_SELECTOR,
    )

    for element in candidates:
        if len(page_elements) >= max_elements:
            max_elements_reached = True
            break

        tag: str | None = None
        property_errors: list[str] = []

        def read_optional(stage: str, operation):
            try:
                return operation()
            except (StaleElementReferenceException, WebDriverException) as error:
                property_errors.append(_exception_summary(stage, error))
                return None

        try:
            tag = _compact_text(element.tag_name, 40).lower()
            if not element.is_displayed():
                if tag == "input":
                    input_collection_statuses.append(
                        (element, "hidden", None, None)
                    )
                continue

            text = _compact_text(
                read_optional("text", lambda: element.text),
                300,
            )
            element_type = _optional_text(
                read_optional("type", lambda: element.get_attribute("type")),
                80,
            )
            name = _optional_text(
                read_optional("name", lambda: element.get_attribute("name"))
            )
            html_id = _optional_text(
                read_optional("id", lambda: element.get_attribute("id"))
            )
            placeholder = _optional_text(
                read_optional(
                    "placeholder",
                    lambda: element.get_attribute("placeholder"),
                )
            )
            aria_label = _optional_text(
                read_optional(
                    "aria-label",
                    lambda: element.get_attribute("aria-label"),
                )
            )
            role = _optional_text(
                read_optional("role", lambda: element.get_attribute("role")),
                80,
            )
            raw_href = _optional_text(
                read_optional("href", lambda: element.get_attribute("href")),
                300,
            )
            href = _safe_url(raw_href, 300) if raw_href else None
            accessible_name = _optional_text(
                read_optional(
                    "accessible_name",
                    lambda: element.accessible_name,
                )
            )
            is_form_control = tag in {"input", "select", "textarea"}
            required = None
            has_value = None
            if is_form_control:
                required_attribute = read_optional(
                    "required",
                    lambda: element.get_attribute("required"),
                )
                aria_required = read_optional(
                    "aria-required",
                    lambda: element.get_attribute("aria-required"),
                )
                required = (
                    required_attribute is not None
                    or aria_required == "true"
                )
                safe_value_types = {
                    None,
                    "email",
                    "number",
                    "search",
                    "tel",
                    "text",
                    "url",
                }
                if tag == "textarea" or (
                    tag == "input" and element_type in safe_value_types
                ):
                    # Only the presence of a value is exposed, never its contents.
                    has_value = bool(
                        read_optional(
                            "value presence",
                            lambda: element.get_attribute("value"),
                        )
                    )

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
                if tag == "input":
                    input_collection_statuses.append(
                        (element, "empty", None, None)
                    )
                continue

            index = len(page_elements) + 1
            enabled = read_optional("is_enabled", element.is_enabled)
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
                    # An unreadable enabled state must never authorize execution.
                    enabled=bool(enabled),
                    in_popup=_is_in_popup(driver, element),
                    required=required,
                    has_value=has_value,
                )
            )
            elements_by_index[index] = element
            if tag == "input":
                input_collection_statuses.append(
                    (
                        element,
                        "included",
                        "; ".join(property_errors) or None,
                        index,
                    )
                )
            if property_errors:
                logger.debug(
                    "Retained element %s with unavailable optional properties: %s",
                    index,
                    "; ".join(property_errors),
                )
        except (StaleElementReferenceException, WebDriverException) as error:
            error_summary = _exception_summary("collection", error)
            if tag == "input":
                input_collection_statuses.append(
                    (element, "error", error_summary, None)
                )
            logger.warning("Skipped an element during page inspection: %s", error_summary)

    observation = PageObservation(
        current_url=_safe_url(driver.current_url),
        title=_compact_text(driver.title, 300),
        elements=page_elements,
        visible_text=_collect_visible_page_text(driver),
    )
    return PageSnapshot(
        observation=observation,
        elements_by_index=elements_by_index,
        input_diagnostics=_diagnose_input_elements(
            driver,
            elements_by_index,
            input_collection_statuses,
            max_elements_reached=max_elements_reached,
        ),
    )


def collect_stable_page_elements(
    driver: WebDriver,
    max_elements: int = 80,
    *,
    stability_timeout: float = 6.0,
    stable_for: float = 1.0,
    poll_interval: float = 0.25,
) -> PageSnapshot:
    """Wait for stable interactive DOM and retry if it changes mid-capture."""
    if stability_timeout <= 0:
        return collect_page_elements(driver, max_elements=max_elements)

    deadline = time.monotonic() + stability_timeout
    attempt = 0
    while True:
        attempt += 1
        remaining = max(0.0, deadline - time.monotonic())
        wait_for_interactive_dom_stable(
            driver,
            timeout=remaining,
            stable_for=stable_for,
            poll_interval=poll_interval,
        )

        signature_before = _useful_element_signature(driver)
        snapshot = collect_page_elements(driver, max_elements=max_elements)
        signature_after = _useful_element_signature(driver)
        if signature_before == signature_after:
            if attempt > 1:
                logger.info("Captured a stable page snapshot on attempt %s", attempt)
            return snapshot

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "Interactive DOM changed during snapshot collection; returning "
                "the latest bounded-time snapshot"
            )
            return snapshot

        logger.info(
            "Interactive DOM changed during snapshot collection; retrying "
            "with %.2f seconds remaining",
            remaining,
        )


def _empty_required_safe_text_fields(snapshot: PageSnapshot) -> list[PageElement]:
    supported_types = {None, "email", "number", "search", "tel", "text", "url"}
    return [
        element
        for element in snapshot.observation.elements
        if element.enabled
        and element.required is True
        and element.has_value is False
        and (
            element.tag == "textarea"
            or (
                element.tag == "input"
                and element.type in supported_types
            )
        )
    ]


def _missing_required_field_before_submit(
    plan: StepPlan,
    snapshot: PageSnapshot,
    confirmation: NextStepConfirmation,
    completed_actions: list[ExecutedAction],
) -> PageElement | None:
    """Return a missing required field when a submit would need invented input."""
    if (
        confirmation.action_type != "click"
        or confirmation.target_element_index is None
        or confirmation.target_element_index not in snapshot.elements_by_index
    ):
        return None

    target = _page_element_by_index(snapshot, confirmation.target_element_index)
    if target.type != "submit":
        return None

    required_fields = _empty_required_safe_text_fields(snapshot)
    if not required_fields:
        return None

    entered_values = {
        action.input_text.casefold()
        for action in completed_actions
        if action.action_type == "type_text" and action.input_text
    }
    unused_provided_values = [
        item.value
        for item in plan.provided_inputs
        if item.value.casefold() not in entered_values
    ]
    if unused_provided_values:
        # The LLM may still be able to bind an unused supplied value to the
        # required field. Its prompt already requires doing so before submit.
        return None
    return required_fields[0]


def _confirmation_consistency_error(
    plan: StepPlan,
    next_step: Step,
    snapshot: PageSnapshot,
    confirmation: NextStepConfirmation,
) -> str | None:
    """Return a correctable error for an internally inconsistent decision."""
    if confirmation.planned_step_number != next_step.step_number:
        return "The confirmation returned the wrong planned step number"

    if confirmation.decision == "skip_ahead":
        if confirmation.action_type != "none":
            return "Skipping plan steps cannot execute a browser action"
        if confirmation.step_progress != "advance_to_next_step":
            return "A skip-ahead decision must advance plan progress"
        if (
            confirmation.next_step_number is None
            or confirmation.next_step_number <= next_step.step_number
            or confirmation.next_step_number > len(plan.steps)
        ):
            return "A skip-ahead decision must identify a valid later plan step"
    elif confirmation.next_step_number is not None:
        return "Only a skip-ahead decision can set next_step_number"

    candidate_indices = [
        candidate.index for candidate in confirmation.candidate_elements
    ]
    if len(candidate_indices) != len(set(candidate_indices)):
        return "The confirmation returned duplicate candidate indexes"

    valid_indices = set(snapshot.elements_by_index)
    reported_indices = set(confirmation.blocking_element_indices) | set(
        candidate_indices
    )
    if confirmation.target_element_index is not None:
        reported_indices.add(confirmation.target_element_index)
    if not reported_indices.issubset(valid_indices):
        return "The confirmation referenced an unknown element index"

    if (
        confirmation.target_element_index is not None
        and confirmation.target_element_index not in candidate_indices
    ):
        return "The selected target was not included in the candidate list"

    if (
        confirmation.action_type in {"click", "type_text"}
        and confirmation.target_element_index is None
    ):
        return "An executable action must include a target element index"
    if confirmation.action_type == "type_text" and not confirmation.input_text:
        return "A type_text action must include input_text"
    if confirmation.action_type != "type_text" and confirmation.input_text is not None:
        return "Only a type_text action can include input_text"

    if (
        confirmation.action_type == "none"
        and confirmation.target_element_index is not None
        and confirmation.decision != "request_input"
    ):
        return "A non-actionable confirmation cannot include a target element"

    if confirmation.decision == "request_input":
        if confirmation.action_type != "none":
            return "A user-input request cannot execute a browser action"
        if confirmation.target_element_index is None:
            return "A user-input request must identify its target field"
        if not confirmation.requested_input:
            return "A user-input request must explain what value is needed"
        requested_element = next(
            (
                element
                for element in snapshot.observation.elements
                if element.index == confirmation.target_element_index
            ),
            None,
        )
        supported_types = {None, "email", "number", "search", "tel", "text", "url"}
        if requested_element is None or not (
            requested_element.required is True
            and requested_element.has_value is False
            and (
                requested_element.tag == "textarea"
                or (
                    requested_element.tag == "input"
                    and requested_element.type in supported_types
                )
            )
        ):
            return (
                "A user-input request must target an empty, required, "
                "safe text field"
            )
    elif confirmation.requested_input is not None:
        return "Only a user-input request can include requested_input"

    if confirmation.decision == "handle_popup" and (
        confirmation.action_type != "click"
        or confirmation.step_progress != "continue_current_step"
    ):
        return "Popup handling must be a click that continues the current step"

    if (
        confirmation.decision not in {"proceed", "handle_popup"}
        and confirmation.action_type != "none"
    ):
        return "A non-actionable decision cannot execute a browser action"

    is_final_step = next_step.step_number == plan.steps[-1].step_number
    if (
        not is_final_step
        and confirmation.decision == "proceed"
        and confirmation.action_type == "none"
        and confirmation.step_progress == "continue_current_step"
    ):
        return (
            "A proceed decision without a browser action must advance a "
            "satisfied step or choose a safe executable action"
        )

    return None


def confirm_next_step(
    plan: StepPlan,
    next_step: Step,
    snapshot: PageSnapshot,
    api_key: str | None = None,
    *,
    last_executed_action: ExecutedAction | None = None,
    completed_actions: list[ExecutedAction] | None = None,
    _rejected_repeated_action: NextStepConfirmation | None = None,
    _rejected_invalid_confirmation: NextStepConfirmation | None = None,
    _rejected_invalid_reason: str | None = None,
) -> NextStepConfirmation:
    """Ask Groq whether the observed page supports the next planned step."""
    prompt_path = Path(__file__).parent / "prompts/confirmnextstep.md"
    system_prompt = prompt_path.read_text(encoding="utf-8")
    page_payload = _observation_prompt_payload(snapshot.observation)
    payload = {
        "normalized_objective": plan.normalized_request,
        "known_ambiguities": plan.ambiguities,
        "provided_inputs": [
            item.model_dump() for item in plan.provided_inputs
        ],
        "success_criteria": plan.success_criteria,
        "planned_next_step": next_step.model_dump(),
        "remaining_plan_steps": [
            step.model_dump()
            for step in plan.steps
            if step.step_number >= next_step.step_number
        ],
        "page_observation": page_payload,
    }
    if last_executed_action is not None:
        action_payload = last_executed_action.model_dump(exclude_none=True)
        action_payload["target_element"] = _element_prompt_payload(
            last_executed_action.target_element
        )
        payload["last_executed_action"] = action_payload

    action_history = list(completed_actions or [])
    if last_executed_action is not None and last_executed_action not in action_history:
        action_history.append(last_executed_action)
    if action_history:
        payload["completed_action_history"] = []
        for action in action_history:
            action_payload = action.model_dump(exclude_none=True)
            action_payload["target_element"] = _element_prompt_payload(
                action.target_element
            )
            payload["completed_action_history"].append(action_payload)
    if _rejected_repeated_action is not None:
        payload["rejected_repeated_action"] = {
            "action_type": _rejected_repeated_action.action_type,
            "target_element_index": (
                _rejected_repeated_action.target_element_index
            ),
            "input_text": _rejected_repeated_action.input_text,
            "reason": (
                "This exact text was already entered into the same populated "
                "field. Choose a different action, such as an applicable "
                "autocomplete option or submit control."
            ),
        }
    if _rejected_invalid_confirmation is not None:
        payload["rejected_confirmation"] = {
            "response": _rejected_invalid_confirmation.model_dump(mode="json"),
            "validation_error": _rejected_invalid_reason,
            "instruction": (
                "Correct the cross-field inconsistency and return a different, "
                "internally consistent decision. Do not repeat the rejected "
                "combination."
            ),
        }

    logger.info(
        "Groq confirmation payload includes %s/%s observed elements",
        page_payload["included_element_count"],
        page_payload["observed_element_count"],
    )

    # Confirmation responses are small. Capping their completion allowance keeps
    # the full request comfortably below Groq's tokens-per-minute calculation.
    llm = initialize_groq_llm(
        api_key=api_key,
        temperature=0.0,
        max_tokens=1_024,
    )
    confirmation = invoke_structured_output(
        llm,
        NextStepConfirmation,
        [
            ("system", system_prompt),
            ("human", json.dumps(payload, ensure_ascii=False)),
        ],
    )

    consistency_error = _confirmation_consistency_error(
        plan,
        next_step,
        snapshot,
        confirmation,
    )
    if consistency_error is not None:
        if _rejected_invalid_confirmation is not None:
            raise ValueError(consistency_error)
        logger.warning(
            "Groq returned an inconsistent confirmation; requesting one "
            "corrected decision: %s",
            consistency_error,
        )
        return confirm_next_step(
            plan,
            next_step,
            snapshot,
            api_key=api_key,
            last_executed_action=last_executed_action,
            completed_actions=action_history,
            _rejected_repeated_action=_rejected_repeated_action,
            _rejected_invalid_confirmation=confirmation,
            _rejected_invalid_reason=consistency_error,
        )

    if confirmation.planned_step_number != next_step.step_number:
        raise ValueError("The confirmation returned the wrong planned step number")
    if confirmation.decision == "skip_ahead":
        if confirmation.action_type != "none":
            raise ValueError("Skipping plan steps cannot execute a browser action")
        if confirmation.step_progress != "advance_to_next_step":
            raise ValueError("A skip-ahead decision must advance plan progress")
        if (
            confirmation.next_step_number is None
            or confirmation.next_step_number <= next_step.step_number
            or confirmation.next_step_number > len(plan.steps)
        ):
            raise ValueError(
                "A skip-ahead decision must identify a valid later plan step"
            )
    elif confirmation.next_step_number is not None:
        raise ValueError("Only a skip-ahead decision can set next_step_number")

    missing_required = _missing_required_field_before_submit(
        plan,
        snapshot,
        confirmation,
        action_history,
    )
    if missing_required is not None:
        field_name = (
            missing_required.accessible_name
            or missing_required.aria_label
            or missing_required.placeholder
            or missing_required.name
            or "the required field"
        )
        logger.warning(
            "Blocked submit action because required field %s is empty",
            missing_required.index,
        )
        confirmation = confirmation.model_copy(
            update={
                "decision": "request_input",
                "confirmed_step_description": (
                    "Collect the missing value required before submission"
                ),
                "action_type": "none",
                "target_element_index": missing_required.index,
                "input_text": None,
                "step_progress": "continue_current_step",
                "candidate_elements": [
                    CandidateEvaluation(
                        index=missing_required.index,
                        reason=(
                            "This visible required field is empty and blocks "
                            "the proposed submit action."
                        ),
                    )
                ],
                "blocking_element_indices": [missing_required.index],
                "expected_result": None,
                "requested_input": f"Enter a value for {field_name}.",
                "next_step_number": None,
                "reason": (
                    "The proposed submission is blocked by an empty required "
                    "safe text field, and no unused user-provided value is "
                    "available for it."
                ),
            }
        )

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
        confirmation.action_type in {"click", "type_text"}
        and confirmation.target_element_index is None
    ):
        raise ValueError("An executable action must include a target element index")
    if confirmation.action_type == "type_text" and not confirmation.input_text:
        raise ValueError("A type_text action must include input_text")
    if (
        confirmation.action_type == "type_text"
        and confirmation.step_progress != "continue_current_step"
    ):
        logger.warning(
            "Groq attempted to advance after text entry; keeping the current "
            "step active for post-entry verification"
        )
        confirmation = confirmation.model_copy(
            update={"step_progress": "continue_current_step"}
        )
    if confirmation.action_type != "type_text" and confirmation.input_text is not None:
        raise ValueError("Only a type_text action can include input_text")
    if confirmation.action_type == "none":
        if (
            confirmation.target_element_index is not None
            and confirmation.decision != "request_input"
        ):
            raise ValueError(
                "A non-actionable confirmation cannot include a target element"
            )
        if (
            confirmation.step_progress != "continue_current_step"
            and confirmation.decision not in {"proceed", "skip_ahead"}
        ):
            logger.warning(
                "Groq attempted to advance without a browser action; keeping "
                "the current step active"
            )
            confirmation = confirmation.model_copy(
                update={"step_progress": "continue_current_step"}
            )
    if confirmation.decision == "request_input":
        if confirmation.action_type != "none":
            raise ValueError("A user-input request cannot execute a browser action")
        if confirmation.target_element_index is None:
            raise ValueError("A user-input request must identify its target field")
        if not confirmation.requested_input:
            raise ValueError("A user-input request must explain what value is needed")
        requested_element = next(
            element
            for element in snapshot.observation.elements
            if element.index == confirmation.target_element_index
        )
        supported_types = {None, "email", "number", "search", "tel", "text", "url"}
        if not (
            requested_element.required is True
            and requested_element.has_value is False
            and (
                requested_element.tag == "textarea"
                or (
                    requested_element.tag == "input"
                    and requested_element.type in supported_types
                )
            )
        ):
            raise ValueError(
                "A user-input request must target an empty, required, safe text field"
            )
    elif confirmation.requested_input is not None:
        raise ValueError("Only a user-input request can include requested_input")
    if confirmation.decision == "handle_popup" and (
        confirmation.action_type != "click"
        or confirmation.step_progress != "continue_current_step"
    ):
        raise ValueError(
            "Popup handling must be a click that continues the current step"
        )
    if (
        confirmation.decision not in {"proceed", "handle_popup"}
        and confirmation.action_type != "none"
    ):
        raise ValueError("A non-actionable decision cannot execute a browser action")

    if _repeats_completed_text_action(
        confirmation,
        snapshot,
        action_history,
    ):
        if _rejected_repeated_action is not None:
            raise ValueError(
                "The decision model repeated text entry into an already "
                "populated field after a corrective retry"
            )
        logger.warning(
            "Groq repeated completed text entry on a populated field; "
            "requesting one corrected action"
        )
        return confirm_next_step(
            plan,
            next_step,
            snapshot,
            api_key=api_key,
            last_executed_action=last_executed_action,
            completed_actions=action_history,
            _rejected_repeated_action=confirmation,
        )

    return confirmation


def extract_final_result(
    plan: StepPlan,
    final_step: Step,
    snapshot: PageSnapshot,
    api_key: str | None = None,
) -> FinalTaskResult:
    """Extract the requested outcome from bounded text on the current page."""
    if final_step != plan.steps[-1]:
        raise ValueError("Final extraction can only run for the last plan step")

    visible_text = getattr(snapshot.observation, "visible_text", "")
    if not visible_text:
        # Some drivers may not expose body.innerText. Element labels are a
        # smaller but still grounded fallback; input values remain excluded.
        visible_text = _bounded_visible_text(
            "\n".join(
                element.text
                or element.accessible_name
                or element.aria_label
                or ""
                for element in snapshot.observation.elements
            )
        )

    prompt_path = Path(__file__).parent / "prompts/extractfinalresult.md"
    system_prompt = prompt_path.read_text(encoding="utf-8")
    payload = {
        "normalized_objective": plan.normalized_request,
        "provided_inputs": [
            item.model_dump() for item in plan.provided_inputs
        ],
        "success_criteria": plan.success_criteria,
        "final_step": final_step.model_dump(),
        "current_url": snapshot.observation.current_url,
        "page_title": snapshot.observation.title,
        "visible_page_text": visible_text,
    }

    llm = initialize_groq_llm(
        api_key=api_key,
        temperature=0.0,
        max_tokens=2_000,
    )
    return invoke_structured_output(
        llm,
        FinalTaskResult,
        [
            ("system", system_prompt),
            ("human", json.dumps(payload, ensure_ascii=False)),
        ],
    )


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
        or page_element.role in {"button", "link", "menuitem", "option", "tab"}
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


def _validate_type_target(
    confirmation: NextStepConfirmation,
    snapshot: PageSnapshot,
) -> tuple[PageElement, WebElement]:
    if confirmation.decision != "proceed":
        raise ValueError(
            f"Decision '{confirmation.decision}' does not authorize text entry"
        )
    if confirmation.action_type != "type_text":
        raise ValueError("The proposed next action is not executable text entry")
    if confirmation.target_element_index is None:
        raise ValueError("The type_text action does not contain a target element")
    if not confirmation.input_text:
        raise ValueError("The type_text action does not contain input text")

    index = confirmation.target_element_index
    page_element = _page_element_by_index(snapshot, index)
    web_element = snapshot.elements_by_index.get(index)
    if web_element is None:
        raise ValueError("The selected element is not part of the latest snapshot")

    supported_input_types = {None, "email", "number", "search", "tel", "text", "url"}
    is_editable_kind = page_element.tag == "textarea" or (
        page_element.tag == "input"
        and page_element.type in supported_input_types
    )
    if not is_editable_kind:
        raise ValueError(
            "Text can only be entered into a supported input or textarea"
        )
    if web_element.get_attribute("readonly") is not None or (
        web_element.get_attribute("aria-readonly") == "true"
    ):
        raise ValueError("The selected text field is read-only")

    return page_element, web_element


def _validate_action_target(
    confirmation: NextStepConfirmation,
    snapshot: PageSnapshot,
) -> tuple[PageElement, WebElement]:
    if confirmation.action_type == "click":
        return _validate_click_target(confirmation, snapshot)
    if confirmation.action_type == "type_text":
        return _validate_type_target(confirmation, snapshot)
    raise ValueError("The proposed next action is not executable")


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


def _repeats_completed_text_action(
    confirmation: NextStepConfirmation,
    snapshot: PageSnapshot,
    completed_actions: list[ExecutedAction],
) -> bool:
    """Detect an exact text replay on the same currently populated field."""
    if (
        confirmation.action_type != "type_text"
        or confirmation.target_element_index is None
        or confirmation.input_text is None
    ):
        return False

    current_target = _page_element_by_index(
        snapshot,
        confirmation.target_element_index,
    )
    if current_target.has_value is not True:
        return False

    current_fingerprint = _element_fingerprint(current_target)
    return any(
        action.action_type == "type_text"
        and action.input_text == confirmation.input_text
        and _element_fingerprint(action.target_element) == current_fingerprint
        for action in completed_actions
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


def _type_text_into_element(
    driver: WebDriver,
    element: WebElement,
    text: str,
    timeout: float,
) -> None:
    """Focus, clear, and type into a validated field using native WebDriver APIs."""
    if not element.is_displayed():
        raise ValueError("The selected text field is no longer visible")
    if not element.is_enabled():
        raise ValueError("The selected text field is disabled")

    driver.execute_script(
        "arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});",
        element,
    )
    editable = WebDriverWait(driver, timeout).until(
        lambda _driver: element
        if element.is_displayed() and element.is_enabled()
        else False
    )
    try:
        editable.click()
        editable.clear()
        editable.send_keys(text)
    except StaleElementReferenceException:
        raise
    except WebDriverException as native_error:
        # Safari can inspect an open-shadow-root descendant but fail its native
        # click command. The fallback remains scoped to validated text fields
        # in open shadow roots and verifies that the complete value was applied.
        fallback_status = driver.execute_script(
            OPEN_SHADOW_TEXT_ENTRY_SCRIPT,
            editable,
            text,
        )
        if fallback_status != "applied":
            logger.warning(
                "Native text entry failed and shadow fallback was unavailable: %s",
                fallback_status,
            )
            raise native_error
        logger.warning(
            "Native text entry failed; applied the approved text through "
            "open-shadow-root input events"
        )


def _perform_confirmed_action(
    driver: WebDriver,
    confirmation: NextStepConfirmation,
    element: WebElement,
    timeout: float,
) -> None:
    if confirmation.action_type == "click":
        _click_element(driver, element, timeout=timeout)
        return
    if confirmation.action_type == "type_text" and confirmation.input_text:
        _type_text_into_element(
            driver,
            element,
            confirmation.input_text,
            timeout=timeout,
        )
        return
    raise ValueError("The confirmation does not contain an executable action")


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
    action_timeout: float = 10.0,
    page_load_timeout: float = 20.0,
    settle_seconds: float = 0.5,
    max_elements: int = 80,
    prepare_next_confirmation: bool = True,
) -> StepRunResult:
    """Execute one confirmed action and optionally prepare the next decision."""
    step_index = _plan_step_index(plan, planned_step)
    if step_index == 0:
        raise ValueError(
            "The navigation step must be executed by navigate_to_plan or "
            "run_first_step"
        )
    if step_index == len(plan.steps) - 1:
        raise ValueError("The final result step is reserved for outcome handling")
    if confirmation.planned_step_number != planned_step.step_number:
        raise ValueError(
            "The proposed action does not belong to the supplied plan step"
        )

    selected_element, web_element = _validate_action_target(
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
        _perform_confirmed_action(
            driver,
            confirmation,
            web_element,
            timeout=action_timeout,
        )
    except StaleElementReferenceException:
        logger.info("Selected element became stale; refreshing the page snapshot")
        fresh_snapshot = collect_stable_page_elements(
            driver,
            max_elements=max_elements,
        )
        remapped = _remap_target(selected_element, fresh_snapshot)

        if remapped is None:
            confirmation = confirm_next_step(
                plan,
                planned_step,
                fresh_snapshot,
                api_key=api_key,
            )
            selected_element, web_element = _validate_action_target(
                confirmation,
                fresh_snapshot,
            )
        else:
            selected_element, web_element = remapped

        pre_action_snapshot = fresh_snapshot
        remapped_after_stale = True
        _perform_confirmed_action(
            driver,
            confirmation,
            web_element,
            timeout=action_timeout,
        )

    current_window_handles = set(driver.window_handles)
    new_window_handles = current_window_handles - previous_window_handles
    if new_window_handles:
        driver.switch_to.window(new_window_handles.pop())

    wait_for_page_ready(driver, timeout=page_load_timeout)
    if settle_seconds > 0:
        time.sleep(settle_seconds)

    post_action_snapshot = collect_stable_page_elements(
        driver,
        max_elements=max_elements,
    )
    page_changed = confirmation.action_type == "type_text" or (
        _observation_signature(pre_action_snapshot.observation)
        != _observation_signature(post_action_snapshot.observation)
    )
    review_step = (
        planned_step
        if confirmation.step_progress == "continue_current_step"
        else plan.steps[step_index + 1]
    )
    executed_action = ExecutedAction(
        planned_step_number=planned_step.step_number,
        action_type=confirmation.action_type,
        target_element=selected_element,
        input_text=confirmation.input_text,
    )
    next_confirmation = None
    if prepare_next_confirmation:
        try:
            next_confirmation = confirm_next_step(
                plan,
                review_step,
                post_action_snapshot,
                api_key=api_key,
                last_executed_action=executed_action,
            )
        except Exception:
            # The browser action has already happened. Preserve its fresh
            # snapshot so callers can retry only the LLM review.
            logger.exception(
                "Could not prepare confirmation for step %s",
                review_step.step_number,
            )

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


def navigate_to_plan(
    driver: WebDriver,
    plan: StepPlan,
    *,
    page_load_timeout: float = 20.0,
    settle_seconds: float = 0.5,
    max_elements: int = 80,
) -> PageSnapshot:
    """Execute the deterministic navigation step and return a fresh snapshot."""
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

    snapshot = collect_stable_page_elements(
        driver,
        max_elements=max_elements,
    )
    logger.info(
        "Completed step %s: current_url=%s observed_elements=%s",
        first_step.step_number,
        snapshot.observation.current_url,
        len(snapshot.observation.elements),
    )
    return snapshot


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
    snapshot = navigate_to_plan(
        driver,
        plan,
        page_load_timeout=page_load_timeout,
        settle_seconds=settle_seconds,
        max_elements=max_elements,
    )

    if len(plan.steps) == 1:
        confirmation = NextStepConfirmation(
            decision="complete",
            planned_step_number=first_step.step_number,
            confirmed_step_description=first_step.description,
            action_type="none",
            target_element_index=None,
            input_text=None,
            step_progress="continue_current_step",
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
