import logging
import os

import streamlit as st
from dotenv import load_dotenv
from selenium import webdriver
from selenium.common.exceptions import WebDriverException

from identifysteps import StepPlan, identify_steps, is_valid_http_url
from runsteps import (
    FirstStepRunResult,
    NextStepConfirmation,
    PageSnapshot,
    StepRunResult,
    confirm_next_step,
    run_first_step,
    run_next_step,
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
    st.session_state.first_step_result = None
    st.session_state.step_results = []


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

    with st.expander("📄 View Raw JSON"):
        st.json(plan.model_dump())


def display_first_step_result(result: FirstStepRunResult) -> None:
    """Render the initial navigation and page observation."""
    st.subheader("🌐 Browser Execution")
    st.success(
        f"Step {result.completed_step.step_number} completed: "
        f"{result.completed_step.description}"
    )

    observation = result.snapshot.observation
    st.markdown(f"**Current URL:** {observation.current_url}")
    st.markdown(f"**Page title:** {observation.title or 'Not available'}")
    st.markdown(f"**Useful elements found:** {len(observation.elements)}")

    with st.expander("🔎 View elements after navigation"):
        if observation.elements:
            st.dataframe(
                [
                    element.model_dump(exclude_none=True)
                    for element in observation.elements
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No visible interactive elements were found on this page.")

def display_step_result(result: StepRunResult) -> None:
    """Render one executed plan step and its fresh page observation."""
    st.markdown(f"#### 🖱️ Step {result.planned_step.step_number} Execution")
    st.success(
        f"Executed one {result.confirmation.action_type} action toward step "
        f"{result.planned_step.step_number}: {result.planned_step.description}"
    )

    selected = result.selected_element
    st.markdown(
        f"**Clicked element:** index {selected.index} — "
        f"{selected.accessible_name or selected.text or selected.tag}"
    )
    if result.remapped_after_stale:
        st.info("The element was safely remapped after the original DOM reference became stale.")

    if result.page_changed:
        st.success("A page-state change was detected after the click.")
    else:
        st.warning(
            "The click completed, but no change was detected in the compact page "
            "observation. The expected result has not been semantically verified yet."
        )

    observation = result.post_action_snapshot.observation
    st.markdown(f"**Current URL:** {observation.current_url}")
    st.markdown(f"**Page title:** {observation.title or 'Not available'}")
    st.markdown(f"**Fresh elements found:** {len(observation.elements)}")

    with st.expander(
        f"🔎 View fresh elements after step {result.planned_step.step_number}"
    ):
        if observation.elements:
            st.dataframe(
                [
                    element.model_dump(exclude_none=True)
                    for element in observation.elements
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No visible interactive elements were found after the click.")


def display_confirmation(
    confirmation: NextStepConfirmation,
    snapshot: PageSnapshot,
    *,
    is_final_step: bool,
) -> None:
    """Render the current proposal against the latest page snapshot."""
    if is_final_step:
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
        if confirmation.target_element_index is not None:
            st.markdown(
                f"**Suggested element index:** {confirmation.target_element_index}"
            )
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


STATE_SCHEMA_VERSION = 4
if st.session_state.get("state_schema_version") != STATE_SCHEMA_VERSION:
    close_browser_session()
    st.session_state.step_plan = None
    st.session_state.state_schema_version = STATE_SCHEMA_VERSION

if "step_plan" not in st.session_state:
    st.session_state.step_plan = None
if "first_step_result" not in st.session_state:
    st.session_state.first_step_result = None
if "step_results" not in st.session_state:
    st.session_state.step_results = []

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
        "Perform Steps executes step 1 and proposes one atomic click for step 2. "
        "Review that proposal before approving the second action."
    )

    perform_column, close_column = st.columns([2, 1])
    with perform_column:
        perform_clicked = st.button(
            "▶️ Perform Steps",
            width="stretch",
            type="primary",
            disabled=st.session_state.first_step_result is not None,
        )
    with close_column:
        close_clicked = st.button(
            "⏹️ Close Browser",
            width="stretch",
            disabled=st.session_state.get("browser_driver") is None,
        )

    if close_clicked:
        close_browser_session()
        st.info("Browser session closed.")

    if perform_clicked:
        if not api_key:
            st.error("❌ A Groq API key is required to review the next step.")
        else:
            with st.spinner("🌐 Opening Safari and performing step 1..."):
                try:
                    driver = st.session_state.get("browser_driver")
                    if driver is None:
                        driver = webdriver.Safari()
                        st.session_state.browser_driver = driver

                    st.session_state.first_step_result = run_first_step(
                        driver=driver,
                        plan=plan,
                        api_key=api_key,
                    )
                except Exception:
                    logger.exception("First-step browser execution failed")
                    close_browser_session()
                    st.error(
                        "❌ Browser execution failed. For Safari, make sure "
                        "Develop → Allow Remote Automation is enabled."
                    )

    first_step_result = st.session_state.first_step_result
    if first_step_result is not None:
        display_first_step_result(first_step_result)

        confirmation = first_step_result.confirmation
        can_execute_step_two = (
            confirmation.decision in {"proceed", "handle_popup"}
            and confirmation.action_type == "click"
            and confirmation.target_element_index is not None
        )

        if can_execute_step_two:
            execute_second_clicked = st.button(
                "🖱️ Approve & Execute Step 2",
                width="stretch",
                type="primary",
                disabled=st.session_state.second_step_result is not None,
            )

            if execute_second_clicked:
                driver = st.session_state.get("browser_driver")
                if driver is None:
                    st.error("❌ The browser session is no longer available.")
                elif not api_key:
                    st.error("❌ A Groq API key is required for stale-page recovery.")
                else:
                    with st.spinner("🖱️ Executing one approved click..."):
                        try:
                            st.session_state.second_step_result = run_second_step(
                                driver=driver,
                                plan=plan,
                                first_step_result=first_step_result,
                                api_key=api_key,
                            )
                        except Exception:
                            logger.exception("Second-step browser execution failed")
                            st.error(
                                "❌ Step 2 could not be executed safely. The browser "
                                "was left open for inspection."
                            )
        else:
            st.info(
                "Step 2 was not offered for execution because Groq did not return "
                "an authorized click action."
            )

    second_step_result = st.session_state.second_step_result
    if second_step_result is not None:
        display_second_step_result(second_step_result)

st.divider()
st.markdown(
    """
    <div style='text-align: center'>
    <p>Built with Streamlit, Groq & Selenium</p>
    </div>
    """,
    unsafe_allow_html=True,
)
