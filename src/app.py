import json
import logging
import os
from typing import Any
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command
from selenium import webdriver
from selenium.common.exceptions import WebDriverException

from graphflow import BrowserRuntime, build_browser_graph, initial_flow_state
from identifysteps import StepPlan, identify_steps, is_valid_http_url
from runsteps import (
    FinalTaskResult,
    NextStepConfirmation,
    PageSnapshot,
)

logger = logging.getLogger(__name__)

load_dotenv()

st.set_page_config(
    page_title="Browser Automation Step Generator",
    page_icon="🤖",
    layout="wide",
)


def close_browser_session() -> None:
    """Close and remove the Selenium browser owned by this Streamlit session."""
    driver = st.session_state.pop("browser_driver", None)
    if driver is not None:
        try:
            driver.quit()
        except WebDriverException:
            logger.warning("The Selenium browser was already unavailable")
    for key in (
        "browser_graph",
        "browser_runtime",
        "graph_config",
        "graph_output",
        "graph_error",
    ):
        st.session_state.pop(key, None)


def display_plan(plan: StepPlan) -> None:
    """Render a generated step plan."""
    with st.container():
        steps_column, summary_column = st.columns([2, 1])

        with steps_column:
            st.subheader("📋 Generated Steps")
            for step in plan.steps:
                with st.expander(f"Step {step.step_number}", expanded=True):
                    st.markdown(step.description)

        with summary_column:
            st.subheader("📊 Task Summary")
            st.markdown(f"**Website:** {plan.url}")
            st.markdown(f"**Original request:** {plan.user_request}")
            st.markdown(f"**Normalized objective:** {plan.normalized_request}")
            st.markdown(f"**Total Steps:** {len(plan.steps)}")
            st.divider()
            st.markdown(f"**Success Criteria:**\n{plan.success_criteria}")

            if plan.ambiguities:
                st.warning("Known ambiguities")
                for ambiguity in plan.ambiguities:
                    st.markdown(f"- {ambiguity}")

            if plan.provided_inputs:
                st.markdown("**Provided inputs:**")
                for provided_input in plan.provided_inputs:
                    st.markdown(
                        f"- {provided_input.label}: `{provided_input.value}`"
                    )

    with st.expander("📄 View Raw JSON"):
        st.json(plan.model_dump())


def display_input_diagnostics(snapshot: PageSnapshot) -> None:
    """Render temporary local diagnostics that are never sent to Groq."""
    diagnostics = snapshot.input_diagnostics
    has_omission = any(
        item.collection_status != "included" for item in diagnostics
    )
    with st.expander(
        "🧪 Temporary input-capture diagnostics",
        expanded=has_omission,
    ):
        st.caption(
            "This table is local debugging information. Input values are not "
            "collected, and this data is not included in the Groq prompt."
        )
        if diagnostics:
            st.dataframe(
                [item.model_dump(exclude_none=True) for item in diagnostics],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Selenium found no top-level input elements in this snapshot.")


def display_confirmation(
    confirmation: NextStepConfirmation,
    snapshot: PageSnapshot,
    *,
    is_final_step: bool,
) -> None:
    """Render the current proposal against the latest page snapshot."""
    if confirmation.decision == "complete":
        st.subheader("✅ Task Complete")
    elif is_final_step:
        st.subheader("🏁 Final Plan Step Reached")
        st.info(
            "Automatic execution pauses before the final result/extraction step. "
            "The final step is shown for review only."
        )
    else:
        st.subheader(f"🧭 Confirm Step {confirmation.planned_step_number}")

    decision_column, page_column = st.columns([1, 1])
    with decision_column:
        if confirmation.decision in {"proceed", "complete"}:
            st.success(f"Decision: {confirmation.decision}")
        else:
            st.warning(f"Decision: {confirmation.decision}")
        st.markdown(f"**Reason:** {confirmation.reason}")
        st.markdown(f"**Proposed action:** {confirmation.action_type}")
        st.markdown(f"**Step progress:** {confirmation.step_progress}")
        if confirmation.next_step_number is not None:
            st.markdown(
                f"**Resume plan at step:** {confirmation.next_step_number}"
            )
        if confirmation.target_element_index is not None:
            st.markdown(
                f"**Suggested element index:** {confirmation.target_element_index}"
            )
        if confirmation.input_text is not None:
            st.markdown("**Text to enter:**")
            st.code(confirmation.input_text, language=None)
        if confirmation.expected_result:
            st.markdown(f"**Expected result:** {confirmation.expected_result}")

    observation = snapshot.observation
    with page_column:
        st.markdown(f"**Current URL:** {observation.current_url}")
        st.markdown(f"**Page title:** {observation.title or 'Not available'}")
        st.markdown(f"**Available elements:** {len(observation.elements)}")

    if confirmation.candidate_elements:
        st.markdown("**Ranked candidate elements**")
        st.dataframe(
            [
                {"rank": rank, **candidate.model_dump()}
                for rank, candidate in enumerate(
                    confirmation.candidate_elements,
                    start=1,
                )
            ],
            width="stretch",
            hide_index=True,
        )

    with st.expander("🧠 View confirmation JSON"):
        st.json(confirmation.model_dump())


def display_final_task_result(
    result: FinalTaskResult,
    snapshot: PageSnapshot,
) -> None:
    """Render the grounded result extracted from the final page."""
    st.subheader("📄 Extracted Result")
    if result.completed:
        st.success(result.summary)
    else:
        st.warning(result.summary)

    if result.details:
        if result.records:
            st.markdown("**Shared details**")
        st.dataframe(
            [detail.model_dump() for detail in result.details],
            width="stretch",
            hide_index=True,
        )
    if result.records:
        st.markdown(f"**Results found:** {len(result.records)}")
        for record in result.records:
            with st.expander(record.title, expanded=True):
                st.dataframe(
                    [detail.model_dump() for detail in record.details],
                    width="stretch",
                    hide_index=True,
                )
    if not result.details and not result.records:
        st.info("No grounded details were found in the captured page text.")

    if result.missing_information:
        st.markdown("**Not available on the captured page:**")
        for item in result.missing_information:
            st.markdown(f"- {item}")

    st.markdown(f"**Source page:** {snapshot.observation.current_url}")
    with st.expander("View extracted result JSON"):
        st.json(result.model_dump())


def current_interrupt_payload(
    graph: Any,
    config: dict[str, Any],
    output: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the active LangGraph interrupt, including after a UI rerun."""
    if output:
        interrupts = output.get("__interrupt__", ())
        if interrupts:
            return interrupts[0].value

    state_snapshot = graph.get_state(config)
    for task in state_snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def display_current_snapshot(snapshot: PageSnapshot) -> None:
    """Render the most recent page observation held by the live runtime."""
    observation = snapshot.observation
    st.subheader("🌐 Current Browser State")
    st.markdown(f"**Current URL:** {observation.current_url}")
    st.markdown(f"**Page title:** {observation.title or 'Not available'}")
    st.markdown(f"**Useful elements found:** {len(observation.elements)}")

    # Temporary snapshot debugging UI is intentionally hidden.
    # with st.expander("🔎 View current elements"):
    #     if observation.elements:
    #         st.dataframe(
    #             [
    #                 element.model_dump(exclude_none=True)
    #                 for element in observation.elements
    #             ],
    #             width="stretch",
    #             hide_index=True,
    #         )
    #     else:
    #         st.info("No visible interactive elements were found on this page.")
    # display_input_diagnostics(snapshot)


def display_action_history(history: list[dict[str, Any]]) -> None:
    """Show every graph-approved tool call without storing WebElements."""
    if not history:
        return
    with st.expander("🧾 Browser action history", expanded=True):
        rows = []
        for item in history:
            row = dict(item)
            row["arguments"] = json.dumps(
                row.get("arguments", {}),
                ensure_ascii=False,
            )
            rows.append(row)
        st.dataframe(rows, width="stretch", hide_index=True)


def resume_browser_graph(
    response: dict[str, Any],
    spinner_text: str,
    *,
    safe_to_retry: bool,
) -> None:
    """Resume the active graph from one human-approval interrupt."""
    graph = st.session_state.get("browser_graph")
    config = st.session_state.get("graph_config")
    if graph is None or config is None:
        st.error("❌ The browser workflow is no longer active.")
        return

    with st.spinner(spinner_text):
        try:
            st.session_state.graph_output = graph.invoke(
                Command(resume=response),
                config,
            )
            st.session_state.graph_error = None
            st.rerun()
        except Exception as error:
            logger.exception("LangGraph browser workflow failed")
            pending_nodes = set(graph.get_state(config).next)
            safe_to_retry = safe_to_retry or bool(
                pending_nodes
                and pending_nodes.isdisjoint({"navigate", "execute_tool"})
            )
            error_message = (
                f"{type(error).__name__}: "
                f"{str(error).strip() or 'No error message was provided.'}"
            )
            st.session_state.graph_error = {
                "message": error_message,
                "safe_to_retry": safe_to_retry,
            }
            st.error(
                "❌ The workflow could not continue safely. The browser was "
                "left open for inspection."
            )
            st.code(error_message, language="text")


def retry_safe_graph_node() -> None:
    """Retry a failed LLM-only/extraction node without replaying a browser tool."""
    graph = st.session_state.get("browser_graph")
    config = st.session_state.get("graph_config")
    if graph is None or config is None:
        st.error("❌ The browser workflow is no longer active.")
        return

    with st.spinner("🔄 Retrying the failed non-browser graph node..."):
        try:
            st.session_state.graph_output = graph.invoke(None, config)
            st.session_state.graph_error = None
            st.rerun()
        except Exception as error:
            logger.exception("Safe LangGraph node retry failed")
            error_message = (
                f"{type(error).__name__}: "
                f"{str(error).strip() or 'No error message was provided.'}"
            )
            st.session_state.graph_error = {
                "message": error_message,
                "safe_to_retry": True,
            }
            st.error("❌ The retry failed. The browser was left unchanged.")
            st.code(error_message, language="text")


def render_interrupt_controls(
    payload: dict[str, Any],
    state: dict[str, Any],
) -> None:
    """Render the human-in-the-loop control for the current graph pause."""
    kind = payload.get("kind")
    step_number = payload.get("step_number", state.get("current_step_number"))
    target_step_number = payload.get("target_step_number", step_number)
    key_suffix = f"{kind}_{step_number}_{len(state.get('action_history', []))}"

    if kind == "input_required":
        context = payload.get("context", {})
        field_context = context.get("field", {})
        current_step = context.get("current_step", {})
        st.warning("A required field is missing and blocks the next submission.")
        st.markdown(
            f"**Current step:** {current_step.get('step_number', step_number)} — "
            f"{current_step.get('description', 'Not available')}"
        )
        st.markdown(
            f"**Required field:** {field_context.get('label', 'Unknown field')}"
        )
        if field_context.get("placeholder"):
            st.markdown(
                f"**Field placeholder:** {field_context['placeholder']}"
            )
        if context.get("request_reason"):
            st.markdown(f"**Why it is needed:** {context['request_reason']}")
        st.caption(
            "Your value will be interpreted using this field context, then shown "
            "to you for review before Selenium enters it."
        )
        supplied_value = st.text_input(
            payload.get("message") or "Enter the required value",
            key=f"graph_input_{key_suffix}",
        )
        if st.button(
            "Use This Value",
            width="stretch",
            type="primary",
            key=f"graph_provide_{key_suffix}",
        ):
            supplied_value = supplied_value.strip()
            if not supplied_value:
                st.error("❌ Enter a value before continuing.")
            elif len(supplied_value) > 500:
                st.error("❌ The supplied value must be 500 characters or fewer.")
            else:
                resume_browser_graph(
                    {"action": "provide", "value": supplied_value},
                    "Preparing the parameterized text-entry tool...",
                    safe_to_retry=True,
                )
        return

    if kind == "input_resolution_approval":
        resolution = payload.get("resolution", {})
        context = payload.get("context", {})
        field_context = context.get("field", {})
        st.subheader("🔎 Review Required Input")
        st.markdown(
            f"**Target field:** {field_context.get('label', 'Required field')}"
        )
        if resolution.get("changed"):
            st.markdown(
                f"**Interpreted value:** `{resolution.get('original_value', '')}` "
                f"→ `{resolution.get('resolved_value', '')}`"
            )
        else:
            st.markdown(
                f"**Validated value:** `{resolution.get('resolved_value', '')}`"
            )
        st.markdown(
            f"**Reason:** {resolution.get('interpretation', 'Not available')}"
        )
        st.markdown(
            f"**Confidence:** {resolution.get('confidence', 'unknown')}"
        )
        if resolution.get("ambiguity"):
            st.warning(f"Ambiguity: {resolution['ambiguity']}")

        accepted_value = st.text_input(
            "Value Selenium will enter",
            value=resolution.get("resolved_value", ""),
            key=f"graph_resolved_input_{key_suffix}",
        )
        if st.button(
            "Approve Resolved Value",
            width="stretch",
            type="primary",
            key=f"graph_resolve_approve_{key_suffix}",
        ):
            accepted_value = accepted_value.strip()
            if not accepted_value:
                st.error("❌ Enter or approve a non-empty value.")
            elif len(accepted_value) > 500:
                st.error("❌ The supplied value must be 500 characters or fewer.")
            else:
                resume_browser_graph(
                    {"action": "approve", "value": accepted_value},
                    "Preparing the approved text-entry tool...",
                    safe_to_retry=True,
                )
        return

    labels = {
        "action_approval": (
            f"▶️ Approve & Execute Step {step_number}",
            "⚙️ Running the approved Selenium tool...",
            "approve",
        ),
        "step_advance_approval": (
            f"➡️ Approve Advance to Step {target_step_number}",
            "🧭 Advancing the tracked plan and reviewing the next step...",
            "approve",
        ),
        "final_extraction_approval": (
            "📄 Extract Final Result",
            "🧠 Extracting grounded details from the final page...",
            "approve",
        ),
        "blocked": (
            f"🔄 Retry Decision for Step {step_number}",
            "🧠 Reviewing the unchanged page snapshot again...",
            "retry",
        ),
    }
    label, spinner, action = labels.get(
        kind,
        ("Continue", "Continuing workflow...", "approve"),
    )
    if payload.get("message"):
        st.info(payload["message"])
    if st.button(
        label,
        width="stretch",
        type="primary",
        key=f"graph_resume_{key_suffix}",
    ):
        resume_browser_graph(
            {"action": action},
            spinner,
            safe_to_retry=(kind != "action_approval"),
        )


STATE_SCHEMA_VERSION = 18
if st.session_state.get("state_schema_version") != STATE_SCHEMA_VERSION:
    close_browser_session()
    st.session_state.step_plan = None
    st.session_state.state_schema_version = STATE_SCHEMA_VERSION

if "step_plan" not in st.session_state:
    st.session_state.step_plan = None
if "graph_error" not in st.session_state:
    st.session_state.graph_error = None

st.title("🤖 Browser Automation Step Generator")
st.markdown(
    "Convert your browser automation requests into high-level step plans using AI"
)

with st.sidebar:
    st.header("Configuration")

    env_api_key = os.getenv("GROQ_API_KEY")
    has_env_key = bool(env_api_key)

    if has_env_key:
        st.success("✅ Groq API key found in .env file")
        use_env_key = st.checkbox("Use API key from .env", value=True)
    else:
        st.warning("⚠️ No Groq API key found in .env file")
        use_env_key = False

    if use_env_key and has_env_key:
        api_key = env_api_key
    else:
        api_key = st.text_input(
            "Enter Groq API Key",
            type="password",
            help="Your Groq API key for LLM access",
        ).strip()

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        """
        This tool uses AI to break down your browser automation task
        into clear, high-level steps.

        **Features:**
        - URL-based planning
        - AI-powered step generation
        - Controlled Selenium execution
        - JSON structured output
        """
    )

st.header("Create Your Step Plan")

with st.form("step_generator_form"):
    url = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        help="Enter the URL of the website you want to automate",
    )

    user_request = st.text_area(
        "Your Task Description",
        placeholder=(
            "e.g., Search for a doctor named Swapna and return their details..."
        ),
        height=120,
        help="Describe what you want to accomplish on the website",
    )

    submitted = st.form_submit_button(
        "✨ Generate Steps",
        width="stretch",
        type="primary",
    )

if submitted:
    if not is_valid_http_url(url.strip()):
        st.error("❌ Please enter a complete HTTP or HTTPS URL")
    elif not user_request.strip():
        st.error("❌ Please describe your task")
    elif not api_key:
        st.error(
            "❌ Groq API key is required. Provide one or add it to your .env file."
        )
    else:
        with st.spinner("🔄 Generating step plan..."):
            try:
                plan = identify_steps(
                    url=url,
                    user_request=user_request,
                    api_key=api_key,
                )
                close_browser_session()
                st.session_state.step_plan = plan
                st.success("✅ Step plan generated successfully!")
            except Exception:
                logger.exception("Step-plan generation failed")
                st.error(
                    "❌ Step generation failed. Review the terminal log and try again."
                )

plan = st.session_state.step_plan
if plan is not None:
    display_plan(plan)

    st.subheader("▶️ Execute Plan")
    st.caption(
        "LangGraph tracks the plan and pauses before every parameterized Selenium "
        "tool call. A fresh page snapshot and LLM decision follow each action."
    )

    perform_column, close_column = st.columns([2, 1])
    with perform_column:
        perform_clicked = st.button(
            "▶️ Perform Steps",
            width="stretch",
            type="primary",
            disabled=st.session_state.get("browser_graph") is not None,
        )
    with close_column:
        close_clicked = st.button(
            "⏹️ Reset Execution",
            width="stretch",
            disabled=st.session_state.get("browser_driver") is None,
        )

    if close_clicked:
        close_browser_session()
        st.rerun()

    if perform_clicked:
        if not api_key:
            st.error("❌ A Groq API key is required to review the next step.")
        else:
            with st.spinner("🌐 Starting the browser graph and performing step 1..."):
                try:
                    driver = st.session_state.get("browser_driver")
                    if driver is None:
                        driver = webdriver.Safari()
                        st.session_state.browser_driver = driver

                    runtime = BrowserRuntime(driver=driver, api_key=api_key)
                    graph = build_browser_graph(runtime)
                    config = {
                        "configurable": {"thread_id": uuid4().hex},
                        "recursion_limit": 100,
                    }
                    st.session_state.browser_runtime = runtime
                    st.session_state.browser_graph = graph
                    st.session_state.graph_config = config
                    st.session_state.graph_output = graph.invoke(
                        initial_flow_state(plan),
                        config,
                    )
                    st.session_state.graph_error = None
                    st.rerun()
                except WebDriverException as error:
                    logger.exception("Browser graph navigation failed")
                    close_browser_session()
                    st.error(
                        "❌ Browser execution failed. For Safari, make sure "
                        "Develop → Allow Remote Automation is enabled."
                    )
                    st.code(
                        f"{type(error).__name__}: "
                        f"{str(error).strip() or 'No browser error message.'}",
                        language="text",
                    )
                except Exception as error:
                    logger.exception("Browser graph startup failed")
                    for key in (
                        "browser_graph",
                        "browser_runtime",
                        "graph_config",
                        "graph_output",
                    ):
                        st.session_state.pop(key, None)
                    st.error(
                        "❌ The graph could not prepare its first decision. The "
                        "Safari session was left open; Perform Steps can retry."
                    )
                    st.code(
                        f"{type(error).__name__}: "
                        f"{str(error).strip() or 'No error message was provided.'}",
                        language="text",
                    )

    graph = st.session_state.get("browser_graph")
    runtime = st.session_state.get("browser_runtime")
    config = st.session_state.get("graph_config")
    if graph is not None and runtime is not None and config is not None:
        graph_snapshot = graph.get_state(config)
        graph_state = dict(graph_snapshot.values)
        snapshot_id = graph_state.get("snapshot_id")
        current_snapshot = runtime.get_snapshot(snapshot_id) if snapshot_id else None

        display_action_history(graph_state.get("action_history", []))
        if current_snapshot is not None:
            display_current_snapshot(current_snapshot)

        confirmation_payload = graph_state.get("confirmation")
        if confirmation_payload is not None and current_snapshot is not None:
            confirmation = NextStepConfirmation.model_validate(
                confirmation_payload
            )
            display_confirmation(
                confirmation,
                current_snapshot,
                is_final_step=(
                    confirmation.planned_step_number == len(plan.steps)
                ),
            )

        final_payload = graph_state.get("final_result")
        if final_payload is not None and current_snapshot is not None:
            display_final_task_result(
                FinalTaskResult.model_validate(final_payload),
                current_snapshot,
            )

        graph_error = st.session_state.get("graph_error")
        if graph_error:
            if graph_error.get("safe_to_retry"):
                st.warning(
                    "The non-browser graph node failed. It is safe to retry; "
                    "Selenium will not repeat the previous action."
                )
                st.code(graph_error["message"], language="text")
                if st.button(
                    "🔄 Retry Failed Graph Node",
                    width="stretch",
                    type="primary",
                ):
                    retry_safe_graph_node()
            else:
                st.error(
                    "The current graph node failed after a browser action was "
                    "approved. No automatic retry was issued because the action "
                    "may already have started. Inspect the browser, then reset."
                )
                st.code(graph_error["message"], language="text")
        elif graph_state.get("status") == "stopped":
            st.warning("The workflow was stopped. Reset execution to start again.")
        elif graph_state.get("status") != "complete":
            interrupt_payload = current_interrupt_payload(
                graph,
                config,
                st.session_state.get("graph_output"),
            )
            if interrupt_payload is not None:
                render_interrupt_controls(interrupt_payload, graph_state)

st.divider()
st.markdown(
    """
    <div style='text-align: center'>
    <p>Built with Streamlit, Groq & Selenium</p>
    </div>
    """,
    unsafe_allow_html=True,
)
