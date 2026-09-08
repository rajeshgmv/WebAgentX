"""LangGraph orchestration for the existing Selenium execution primitives.

Only JSON-serializable task state is checkpointed.  The live WebDriver,
WebElements, and PageSnapshot objects remain in BrowserRuntime because those
objects are tied to the current browser process and cannot be safely restored
from a graph checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from selenium.webdriver.remote.webdriver import WebDriver

if __package__:
    from .identifysteps import Step, StepPlan
    from .resolveinput import InputResolution, resolve_required_input
    from .rewriteuserrequest import ProvidedInput
    from .runsteps import (
        ExecutedAction,
        NextStepConfirmation,
        PageElement,
        PageSnapshot,
        confirm_next_step,
        extract_final_result,
        navigate_to_plan,
        run_next_step,
    )
else:
    from identifysteps import Step, StepPlan
    from resolveinput import InputResolution, resolve_required_input
    from rewriteuserrequest import ProvidedInput
    from runsteps import (
        ExecutedAction,
        NextStepConfirmation,
        PageElement,
        PageSnapshot,
        confirm_next_step,
        extract_final_result,
        navigate_to_plan,
        run_next_step,
    )


FlowStatus = Literal[
    "ready",
    "running",
    "complete",
    "stopped",
]


class BrowserFlowState(TypedDict, total=False):
    """Serializable state owned by LangGraph's checkpointer."""

    plan: dict[str, Any]
    current_step_number: int
    snapshot_id: str
    observation: dict[str, Any]
    confirmation: dict[str, Any] | None
    pending_tool_call: dict[str, Any] | None
    last_executed_action: dict[str, Any] | None
    completed_actions: list[dict[str, Any]]
    pending_input: dict[str, Any] | None
    input_resolution_history: list[dict[str, Any]]
    action_history: list[dict[str, Any]]
    status: FlowStatus
    final_result: dict[str, Any] | None
    error: str | None


@dataclass(slots=True)
class BrowserRuntime:
    """Non-serializable resources for one active browser graph."""

    driver: WebDriver
    api_key: str | None
    snapshots: dict[str, PageSnapshot] = field(default_factory=dict)

    def store_snapshot(self, snapshot: PageSnapshot) -> str:
        snapshot_id = uuid4().hex
        self.snapshots[snapshot_id] = snapshot
        return snapshot_id

    def get_snapshot(self, snapshot_id: str) -> PageSnapshot:
        try:
            return self.snapshots[snapshot_id]
        except KeyError as error:
            raise RuntimeError(
                "The live page snapshot is no longer available. Start a new run."
            ) from error


def initial_flow_state(plan: StepPlan) -> BrowserFlowState:
    """Create a fresh serializable state for one generated plan."""
    return BrowserFlowState(
        plan=plan.model_dump(mode="json"),
        current_step_number=1,
        confirmation=None,
        pending_tool_call=None,
        last_executed_action=None,
        completed_actions=[],
        pending_input=None,
        input_resolution_history=[],
        action_history=[],
        status="ready",
        final_result=None,
        error=None,
    )


def _plan(state: BrowserFlowState) -> StepPlan:
    return StepPlan.model_validate(state["plan"])


def _step(plan: StepPlan, step_number: int) -> Step:
    if not 1 <= step_number <= len(plan.steps):
        raise ValueError(f"Step {step_number} is not present in the plan")
    return plan.steps[step_number - 1]


def _confirmation(state: BrowserFlowState) -> NextStepConfirmation:
    payload = state.get("confirmation")
    if payload is None:
        raise ValueError("No current LLM confirmation is available")
    return NextStepConfirmation.model_validate(payload)


def _last_action(state: BrowserFlowState) -> ExecutedAction | None:
    payload = state.get("last_executed_action")
    return ExecutedAction.model_validate(payload) if payload else None


def _completed_actions(state: BrowserFlowState) -> list[ExecutedAction]:
    return [
        ExecutedAction.model_validate(payload)
        for payload in state.get("completed_actions", [])
    ]


def _snapshot(runtime: BrowserRuntime, state: BrowserFlowState) -> PageSnapshot:
    snapshot_id = state.get("snapshot_id")
    if not snapshot_id:
        raise RuntimeError("No current page snapshot is available")
    return runtime.get_snapshot(snapshot_id)


def _tool_call(confirmation: NextStepConfirmation) -> dict[str, Any] | None:
    """Translate the validated model decision into a bounded browser tool call."""
    if confirmation.action_type not in {"click", "type_text"}:
        return None

    arguments: dict[str, Any] = {
        "element_index": confirmation.target_element_index,
    }
    if confirmation.action_type == "type_text":
        arguments["text"] = confirmation.input_text
    return {
        "name": confirmation.action_type,
        "arguments": arguments,
        "planned_step_number": confirmation.planned_step_number,
    }


def _decision_update(
    confirmation: NextStepConfirmation,
) -> BrowserFlowState:
    return BrowserFlowState(
        confirmation=confirmation.model_dump(mode="json"),
        pending_tool_call=_tool_call(confirmation),
        current_step_number=confirmation.planned_step_number or 1,
        error=None,
    )


def _response_action(response: object) -> str:
    if isinstance(response, dict):
        return str(response.get("action", "")).strip().casefold()
    if isinstance(response, str):
        return response.strip().casefold()
    return ""


def _is_approved(response: object) -> bool:
    return _response_action(response) in {"approve", "approved", "yes", "continue"}


def _target_element(snapshot: PageSnapshot, index: int | None) -> PageElement:
    if index is None:
        raise ValueError("The required input has no target element")
    for element in snapshot.observation.elements:
        if element.index == index:
            return element
    raise ValueError(f"Element index {index} is not in the current snapshot")


def _input_label(element: PageElement) -> str:
    return (
        element.accessible_name
        or element.aria_label
        or element.placeholder
        or element.name
        or "required input"
    )[:100]


def _append_provided_input(
    plan: StepPlan,
    element: PageElement,
    value: str,
) -> StepPlan:
    supplied = ProvidedInput(label=_input_label(element), value=value)
    existing = {
        (item.label.casefold(), item.value.casefold()) for item in plan.provided_inputs
    }
    if (supplied.label.casefold(), supplied.value.casefold()) in existing:
        return plan
    return plan.model_copy(
        update={"provided_inputs": [*plan.provided_inputs, supplied]}
    )


def _route_confirmation(state: BrowserFlowState) -> str:
    payload = state.get("confirmation")
    if payload is None:
        raise ValueError("The LLM decision node did not produce a confirmation")

    confirmation = NextStepConfirmation.model_validate(payload)
    plan = _plan(state)
    step_number = confirmation.planned_step_number or state["current_step_number"]
    is_final_step = step_number == len(plan.steps)

    if confirmation.decision == "request_input":
        return "request_input"
    if confirmation.decision == "skip_ahead":
        return "review_advance"
    if confirmation.action_type in {"click", "type_text"}:
        if (
            not is_final_step
            and confirmation.decision in {"proceed", "handle_popup"}
        ):
            return "review_action"
        return "review_blocked"
    if confirmation.action_type == "none" and (
        is_final_step or confirmation.decision == "complete"
    ):
        return "review_final"
    if (
        confirmation.decision == "proceed"
        and confirmation.action_type == "none"
        and confirmation.step_progress == "advance_to_next_step"
    ):
        return "review_advance"
    return "review_blocked"


def build_browser_graph(
    runtime: BrowserRuntime,
    *,
    checkpointer: InMemorySaver | None = None,
):
    """Compile the human-approved browser workflow around existing logic."""

    def navigate(state: BrowserFlowState) -> BrowserFlowState:
        plan = _plan(state)
        snapshot = navigate_to_plan(
            driver=runtime.driver,
            plan=plan,
        )
        snapshot_id = runtime.store_snapshot(snapshot)
        history = [
            *state.get("action_history", []),
            {
                "sequence": len(state.get("action_history", [])) + 1,
                "planned_step_number": plan.steps[0].step_number,
                "tool": "navigate",
                "arguments": {"url": plan.url},
                "status": "completed",
                "url_after": snapshot.observation.current_url,
            },
        ]
        return BrowserFlowState(
            snapshot_id=snapshot_id,
            observation=snapshot.observation.model_dump(mode="json"),
            confirmation=None,
            pending_tool_call=None,
            current_step_number=2 if len(plan.steps) > 1 else 1,
            action_history=history,
            status="running",
            error=None,
        )

    def decide(state: BrowserFlowState) -> BrowserFlowState:
        plan = _plan(state)
        step = _step(plan, state["current_step_number"])
        if len(plan.steps) == 1:
            confirmation = NextStepConfirmation(
                decision="complete",
                planned_step_number=step.step_number,
                confirmed_step_description=step.description,
                action_type="none",
                target_element_index=None,
                input_text=None,
                step_progress="continue_current_step",
                candidate_elements=[],
                blocking_element_indices=[],
                expected_result=None,
                requested_input=None,
                reason="The generated plan contains no additional steps.",
            )
        else:
            confirmation = confirm_next_step(
                plan,
                step,
                _snapshot(runtime, state),
                api_key=runtime.api_key,
                last_executed_action=_last_action(state),
                completed_actions=_completed_actions(state),
            )
        return _decision_update(confirmation)

    def review_action(state: BrowserFlowState) -> BrowserFlowState:
        response = interrupt(
            {
                "kind": "action_approval",
                "message": "Approve this browser action before Selenium runs it.",
                "step_number": state["current_step_number"],
                "confirmation": state["confirmation"],
                "tool_call": state.get("pending_tool_call"),
            }
        )
        return BrowserFlowState(
            status="running" if _is_approved(response) else "stopped"
        )

    def after_action_review(state: BrowserFlowState) -> str:
        return "execute_tool" if state["status"] == "running" else "end"

    def execute_tool(state: BrowserFlowState) -> BrowserFlowState:
        plan = _plan(state)
        confirmation = _confirmation(state)
        planned_step = _step(plan, state["current_step_number"])
        before = _snapshot(runtime, state)
        result = run_next_step(
            driver=runtime.driver,
            plan=plan,
            planned_step=planned_step,
            pre_action_snapshot=before,
            confirmation=confirmation,
            api_key=runtime.api_key,
            prepare_next_confirmation=False,
        )
        snapshot_id = runtime.store_snapshot(result.post_action_snapshot)
        executed_action = result.executed_action
        completed_actions = [
            *state.get("completed_actions", []),
            executed_action.model_dump(mode="json"),
        ]
        next_step_number = (
            planned_step.step_number
            if confirmation.step_progress == "continue_current_step"
            else planned_step.step_number + 1
        )
        history = [
            *state.get("action_history", []),
            {
                "sequence": len(state.get("action_history", [])) + 1,
                "planned_step_number": planned_step.step_number,
                "tool": confirmation.action_type,
                "arguments": (state.get("pending_tool_call") or {}).get(
                    "arguments", {}
                ),
                "status": "completed",
                "url_before": before.observation.current_url,
                "url_after": result.post_action_snapshot.observation.current_url,
                "page_changed": result.page_changed,
                "remapped_after_stale": result.remapped_after_stale,
            },
        ]
        return BrowserFlowState(
            snapshot_id=snapshot_id,
            observation=result.post_action_snapshot.observation.model_dump(
                mode="json"
            ),
            confirmation=None,
            pending_tool_call=None,
            current_step_number=next_step_number,
            last_executed_action=executed_action.model_dump(mode="json"),
            completed_actions=completed_actions,
            action_history=history,
            status="running",
            error=None,
        )

    def request_input(state: BrowserFlowState) -> BrowserFlowState:
        plan = _plan(state)
        confirmation = _confirmation(state)
        snapshot = _snapshot(runtime, state)
        target = _target_element(snapshot, confirmation.target_element_index)
        step = _step(plan, state["current_step_number"])
        input_context = {
            "normalized_objective": plan.normalized_request,
            "current_step": step.model_dump(mode="json"),
            "request_reason": confirmation.reason,
            "requested_input": confirmation.requested_input,
            "page": {
                "url": snapshot.observation.current_url,
                "title": snapshot.observation.title,
            },
            "field": {
                "label": _input_label(target),
                "tag": target.tag,
                "type": target.type,
                "name": target.name,
                "placeholder": target.placeholder,
                "aria_label": target.aria_label,
                "accessible_name": target.accessible_name,
                "required": target.required,
            },
        }
        response = interrupt(
            {
                "kind": "input_required",
                "message": confirmation.requested_input
                or "Enter the required value.",
                "step_number": state["current_step_number"],
                "target_element": target.model_dump(mode="json"),
                "context": input_context,
            }
        )
        if _response_action(response) in {"cancel", "stop", "reject"}:
            return BrowserFlowState(status="stopped")

        value = response.get("value") if isinstance(response, dict) else response
        value = str(value or "").strip()
        if not value:
            raise ValueError("A non-empty value is required")
        if len(value) > 500:
            raise ValueError("The supplied value must be 500 characters or fewer")

        return BrowserFlowState(
            pending_input={
                "original_value": value,
                "target_element_index": target.index,
                "context": input_context,
            },
            status="running",
            error=None,
        )

    def resolve_input(state: BrowserFlowState) -> BrowserFlowState:
        pending_input = state.get("pending_input")
        if pending_input is None:
            raise ValueError("No required input is waiting to be resolved")
        resolution = resolve_required_input(
            pending_input["original_value"],
            pending_input["context"],
            api_key=runtime.api_key,
        )
        return BrowserFlowState(
            pending_input={
                **pending_input,
                "resolution": resolution.model_dump(mode="json"),
            },
            status="running",
            error=None,
        )

    def review_input_resolution(state: BrowserFlowState) -> BrowserFlowState:
        pending_input = state.get("pending_input")
        if pending_input is None or "resolution" not in pending_input:
            raise ValueError("No resolved required input is available for review")
        resolution = InputResolution.model_validate(pending_input["resolution"])
        response = interrupt(
            {
                "kind": "input_resolution_approval",
                "message": (
                    "Review the field-aware value before Selenium enters it."
                ),
                "step_number": state["current_step_number"],
                "context": pending_input["context"],
                "resolution": resolution.model_dump(mode="json"),
            }
        )
        if _response_action(response) in {"cancel", "stop", "reject"}:
            return BrowserFlowState(status="stopped")

        accepted_value = (
            response.get("value") if isinstance(response, dict) else None
        )
        accepted_value = str(accepted_value or resolution.resolved_value).strip()
        if not accepted_value:
            raise ValueError("The approved required input cannot be empty")
        if len(accepted_value) > 500:
            raise ValueError("The approved required input is too long")

        confirmation = _confirmation(state)
        snapshot = _snapshot(runtime, state)
        target = _target_element(
            snapshot,
            pending_input["target_element_index"],
        )
        updated_confirmation = confirmation.model_copy(
            update={
                "decision": "proceed",
                "confirmed_step_description": (
                    "Enter the user-provided value into the required field"
                ),
                "action_type": "type_text",
                "input_text": accepted_value,
                "step_progress": "continue_current_step",
                "expected_result": (
                    "The required field contains the user-provided value."
                ),
                "requested_input": None,
                "reason": (
                    "The user reviewed the field-aware interpretation of the "
                    "missing value; it can now be entered into the selected field."
                ),
            }
        )
        plan = _append_provided_input(_plan(state), target, accepted_value)
        resolution_record = {
            **resolution.model_dump(mode="json"),
            "accepted_value": accepted_value,
            "step_number": state["current_step_number"],
            "target_element": target.model_dump(mode="json"),
        }
        return BrowserFlowState(
            plan=plan.model_dump(mode="json"),
            confirmation=updated_confirmation.model_dump(mode="json"),
            pending_tool_call=_tool_call(updated_confirmation),
            pending_input=None,
            input_resolution_history=[
                *state.get("input_resolution_history", []),
                resolution_record,
            ],
            status="running",
            error=None,
        )

    def after_input(state: BrowserFlowState) -> str:
        return "resolve_input" if state["status"] == "running" else "end"

    def after_input_resolution(state: BrowserFlowState) -> str:
        return "review_action" if state["status"] == "running" else "end"

    def review_advance(state: BrowserFlowState) -> BrowserFlowState:
        confirmation = _confirmation(state)
        target_step_number = (
            confirmation.next_step_number
            if confirmation.decision == "skip_ahead"
            else state["current_step_number"] + 1
        )
        response = interrupt(
            {
                "kind": "step_advance_approval",
                "message": (
                    "The current page indicates that no browser action is needed "
                    "for the skipped plan work. Approve updating tracked progress "
                    f"to step {target_step_number} without changing the page."
                ),
                "step_number": state["current_step_number"],
                "target_step_number": target_step_number,
                "confirmation": state["confirmation"],
            }
        )
        return BrowserFlowState(
            status="running" if _is_approved(response) else "stopped"
        )

    def after_advance_review(state: BrowserFlowState) -> str:
        return "advance_step" if state["status"] == "running" else "end"

    def advance_step(state: BrowserFlowState) -> BrowserFlowState:
        plan = _plan(state)
        confirmation = _confirmation(state)
        next_step_number = (
            confirmation.next_step_number
            if confirmation.decision == "skip_ahead"
            else state["current_step_number"] + 1
        )
        if next_step_number is None:
            raise ValueError("The approved step advance has no target step")
        if next_step_number > len(plan.steps):
            next_step_number = len(plan.steps)
        history = [
            *state.get("action_history", []),
            {
                "sequence": len(state.get("action_history", [])) + 1,
                "planned_step_number": state["current_step_number"],
                "tool": "advance_step",
                "arguments": {
                    "from_step_number": state["current_step_number"],
                    "to_step_number": next_step_number,
                },
                "status": "completed",
                "url_after": state["observation"]["current_url"],
            },
        ]
        return BrowserFlowState(
            current_step_number=next_step_number,
            confirmation=None,
            pending_tool_call=None,
            action_history=history,
            status="running",
            error=None,
        )

    def review_final(state: BrowserFlowState) -> BrowserFlowState:
        response = interrupt(
            {
                "kind": "final_extraction_approval",
                "message": (
                    "Approve sending bounded visible page text to Groq for "
                    "grounded final-result extraction."
                ),
                "step_number": state["current_step_number"],
                "confirmation": state["confirmation"],
            }
        )
        return BrowserFlowState(
            status="running" if _is_approved(response) else "stopped"
        )

    def after_final_review(state: BrowserFlowState) -> str:
        return "extract_final" if state["status"] == "running" else "end"

    def extract_final(state: BrowserFlowState) -> BrowserFlowState:
        plan = _plan(state)
        result = extract_final_result(
            plan,
            plan.steps[-1],
            _snapshot(runtime, state),
            api_key=runtime.api_key,
        )
        history = [
            *state.get("action_history", []),
            {
                "sequence": len(state.get("action_history", [])) + 1,
                "planned_step_number": plan.steps[-1].step_number,
                "tool": "extract_final_result",
                "arguments": {},
                "status": "completed",
                "url_after": state["observation"]["current_url"],
            },
        ]
        return BrowserFlowState(
            final_result=result.model_dump(mode="json"),
            action_history=history,
            status="complete",
            pending_tool_call=None,
            error=None,
        )

    def review_blocked(state: BrowserFlowState) -> BrowserFlowState:
        response = interrupt(
            {
                "kind": "blocked",
                "message": (
                    "The model did not produce a safe executable action. "
                    "You may retry the decision against the unchanged snapshot."
                ),
                "step_number": state["current_step_number"],
                "confirmation": state["confirmation"],
            }
        )
        retry = _response_action(response) in {"retry", "approve", "continue"}
        return BrowserFlowState(
            status="running" if retry else "stopped",
            confirmation=None if retry else state.get("confirmation"),
            pending_tool_call=None,
        )

    def after_blocked(state: BrowserFlowState) -> str:
        return "decide" if state["status"] == "running" else "end"

    graph = StateGraph(BrowserFlowState)
    graph.add_node("navigate", navigate)
    graph.add_node("decide", decide)
    graph.add_node("review_action", review_action)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("request_input", request_input)
    graph.add_node("resolve_input", resolve_input)
    graph.add_node("review_input_resolution", review_input_resolution)
    graph.add_node("review_advance", review_advance)
    graph.add_node("advance_step", advance_step)
    graph.add_node("review_final", review_final)
    graph.add_node("extract_final", extract_final)
    graph.add_node("review_blocked", review_blocked)

    decision_routes = {
        "review_action": "review_action",
        "request_input": "request_input",
        "review_advance": "review_advance",
        "review_final": "review_final",
        "review_blocked": "review_blocked",
    }
    graph.add_edge(START, "navigate")
    graph.add_edge("navigate", "decide")
    graph.add_conditional_edges(
        "decide",
        _route_confirmation,
        decision_routes,
    )
    graph.add_conditional_edges(
        "review_action",
        after_action_review,
        {"execute_tool": "execute_tool", "end": END},
    )
    graph.add_edge("execute_tool", "decide")
    graph.add_conditional_edges(
        "request_input",
        after_input,
        {"resolve_input": "resolve_input", "end": END},
    )
    graph.add_edge("resolve_input", "review_input_resolution")
    graph.add_conditional_edges(
        "review_input_resolution",
        after_input_resolution,
        {"review_action": "review_action", "end": END},
    )
    graph.add_conditional_edges(
        "review_advance",
        after_advance_review,
        {"advance_step": "advance_step", "end": END},
    )
    graph.add_edge("advance_step", "decide")
    graph.add_conditional_edges(
        "review_final",
        after_final_review,
        {"extract_final": "extract_final", "end": END},
    )
    graph.add_edge("extract_final", END)
    graph.add_conditional_edges(
        "review_blocked",
        after_blocked,
        {"decide": "decide", "end": END},
    )

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
