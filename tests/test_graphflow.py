import json
import unittest
from unittest.mock import patch

from langgraph.types import Command

from src.graphflow import BrowserRuntime, build_browser_graph, initial_flow_state
from src.identifysteps import Step, StepPlan
from src.resolveinput import InputResolution
from src.rewriteuserrequest import ProvidedInput
from src.runsteps import (
    CandidateEvaluation,
    ExtractedDetail,
    FinalTaskResult,
    NextStepConfirmation,
    PageElement,
    PageObservation,
    PageSnapshot,
    StepRunResult,
)


def make_plan() -> StepPlan:
    return StepPlan(
        url="https://example.com",
        user_request="Find product Atlas in Chicago",
        normalized_request="Find product Atlas in Chicago and return its details",
        ambiguities=[],
        provided_inputs=[
            ProvidedInput(label="product name", value="Atlas"),
            ProvidedInput(label="location", value="Chicago"),
        ],
        steps=[
            Step(step_number=1, description="Navigate to the site"),
            Step(step_number=2, description="Find the matching product"),
            Step(step_number=3, description="Extract the product details"),
        ],
        success_criteria="The matching product details are returned",
    )


def make_snapshot(*, required: bool = False) -> PageSnapshot:
    page_element = PageElement(
        index=1,
        tag="input" if required else "a",
        text="" if required else "Find products",
        type="text" if required else None,
        name="location" if required else None,
        id=None,
        placeholder="Near (required)" if required else None,
        aria_label="Location" if required else None,
        accessible_name="Location" if required else "Find products",
        role=None if required else "link",
        href=None if required else "https://example.com/products",
        enabled=True,
        in_popup=False,
        required=required,
        has_value=False if required else None,
    )
    return PageSnapshot(
        observation=PageObservation(
            current_url="https://example.com",
            title="Example",
            elements=[page_element],
            visible_text="Atlas product details",
        ),
        elements_by_index={1: object()},
    )


def make_click_confirmation() -> NextStepConfirmation:
    return NextStepConfirmation(
        decision="proceed",
        planned_step_number=2,
        confirmed_step_description="Open the matching workflow",
        action_type="click",
        target_element_index=1,
        input_text=None,
        step_progress="advance_to_next_step",
        candidate_elements=[
            CandidateEvaluation(index=1, reason="Matches the complete objective")
        ],
        blocking_element_indices=[],
        expected_result="The matching result opens",
        requested_input=None,
        reason="The link is the best match for the complete objective",
    )


def make_input_confirmation() -> NextStepConfirmation:
    return NextStepConfirmation(
        decision="request_input",
        planned_step_number=2,
        confirmed_step_description="Supply the missing location",
        action_type="none",
        target_element_index=1,
        input_text=None,
        step_progress="continue_current_step",
        candidate_elements=[
            CandidateEvaluation(index=1, reason="Empty required location field")
        ],
        blocking_element_indices=[],
        expected_result=None,
        requested_input="Enter the required location",
        reason="A required location value was not supplied",
    )


def make_final_confirmation() -> NextStepConfirmation:
    return NextStepConfirmation(
        decision="complete",
        planned_step_number=3,
        confirmed_step_description="Extract the product details",
        action_type="none",
        target_element_index=None,
        input_text=None,
        step_progress="continue_current_step",
        candidate_elements=[],
        blocking_element_indices=[],
        expected_result=None,
        requested_input=None,
        reason="The requested details are visible",
    )


class GraphFlowTests(unittest.TestCase):
    def test_skip_ahead_updates_current_step_without_browser_action(self):
        base_plan = make_plan()
        plan = base_plan.model_copy(
            update={
                "steps": [
                    Step(step_number=1, description="Navigate"),
                    Step(step_number=2, description="Search"),
                    Step(step_number=3, description="Filter"),
                    Step(step_number=4, description="Open matching result"),
                    Step(step_number=5, description="Extract details"),
                ]
            }
        )
        snapshot = make_snapshot()
        skip = NextStepConfirmation(
            decision="skip_ahead",
            planned_step_number=2,
            confirmed_step_description="Search and filtering are already complete",
            action_type="none",
            target_element_index=None,
            input_text=None,
            step_progress="advance_to_next_step",
            candidate_elements=[],
            blocking_element_indices=[],
            expected_result=None,
            requested_input=None,
            next_step_number=4,
            reason="The current results page proves steps 2 and 3 are complete.",
        )
        step_four_review = NextStepConfirmation(
            decision="replan",
            planned_step_number=4,
            confirmed_step_description=None,
            action_type="none",
            target_element_index=None,
            input_text=None,
            step_progress="continue_current_step",
            candidate_elements=[],
            blocking_element_indices=[],
            expected_result=None,
            requested_input=None,
            reason="Test pause after the jump.",
        )
        runtime = BrowserRuntime(driver=object(), api_key="test-key")
        config = {"configurable": {"thread_id": "skip-flow"}}

        with (
            patch("src.graphflow.navigate_to_plan", return_value=snapshot),
            patch(
                "src.graphflow.confirm_next_step",
                side_effect=[skip, step_four_review],
            ),
            patch("src.graphflow.run_next_step") as run,
        ):
            graph = build_browser_graph(runtime)
            approval = graph.invoke(initial_flow_state(plan), config)
            self.assertEqual(
                approval["__interrupt__"][0].value["target_step_number"],
                4,
            )
            paused = graph.invoke(Command(resume={"action": "approve"}), config)

        self.assertEqual(paused["__interrupt__"][0].value["kind"], "blocked")
        state = graph.get_state(config).values
        self.assertEqual(state["current_step_number"], 4)
        self.assertEqual(
            state["action_history"][-1]["arguments"],
            {"from_step_number": 2, "to_step_number": 4},
        )
        run.assert_not_called()

    def test_approved_action_reaches_final_extraction_and_tracks_tools(self):
        plan = make_plan()
        before = make_snapshot()
        after = make_snapshot()
        click_confirmation = make_click_confirmation()
        final_confirmation = make_final_confirmation()
        step_result = StepRunResult(
            planned_step=plan.steps[1],
            confirmation=click_confirmation,
            selected_element=before.observation.elements[0],
            pre_action_snapshot=before,
            post_action_snapshot=after,
            page_changed=True,
            remapped_after_stale=False,
            next_confirmation=final_confirmation,
        )
        final_result = FinalTaskResult(
            completed=True,
            summary="Atlas was found.",
            details=[ExtractedDetail(label="Name", value="Atlas")],
            missing_information=[],
        )
        runtime = BrowserRuntime(driver=object(), api_key="test-key")
        config = {"configurable": {"thread_id": "approved-flow"}}

        with (
            patch(
                "src.graphflow.navigate_to_plan",
                return_value=before,
            ),
            patch(
                "src.graphflow.confirm_next_step",
                side_effect=[click_confirmation, final_confirmation],
            ),
            patch("src.graphflow.run_next_step", return_value=step_result) as run,
            patch(
                "src.graphflow.extract_final_result",
                return_value=final_result,
            ) as extract,
        ):
            graph = build_browser_graph(runtime)
            first_pause = graph.invoke(initial_flow_state(plan), config)
            self.assertEqual(first_pause["__interrupt__"][0].value["kind"], "action_approval")

            # WebDriver and WebElement objects never enter checkpointed state.
            json.dumps(graph.get_state(config).values)

            final_pause = graph.invoke(Command(resume={"action": "approve"}), config)
            self.assertEqual(
                final_pause["__interrupt__"][0].value["kind"],
                "final_extraction_approval",
            )
            completed = graph.invoke(Command(resume={"action": "approve"}), config)

        run.assert_called_once()
        extract.assert_called_once()
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(completed["final_result"]["summary"], "Atlas was found.")
        self.assertEqual(
            [item["tool"] for item in completed["action_history"]],
            ["navigate", "click", "extract_final_result"],
        )

    def test_required_input_becomes_an_approved_type_text_tool(self):
        plan = make_plan()
        snapshot = make_snapshot(required=True)
        confirmation = make_input_confirmation()
        runtime = BrowserRuntime(driver=object(), api_key="test-key")
        config = {"configurable": {"thread_id": "input-flow"}}

        with patch(
            "src.graphflow.navigate_to_plan",
            return_value=snapshot,
        ), patch(
            "src.graphflow.confirm_next_step",
            return_value=confirmation,
        ), patch(
            "src.graphflow.resolve_required_input",
            return_value=InputResolution(
                original_value="MKE",
                resolved_value="Milwaukee",
                changed=True,
                interpretation="MKE identifies Milwaukee in a location field.",
                confidence="high",
                ambiguity=None,
            ),
        ):
            graph = build_browser_graph(runtime)
            first_pause = graph.invoke(initial_flow_state(plan), config)
            self.assertEqual(first_pause["__interrupt__"][0].value["kind"], "input_required")

            resolution_pause = graph.invoke(
                Command(resume={"action": "provide", "value": "MKE"}),
                config,
            )
            self.assertEqual(
                resolution_pause["__interrupt__"][0].value["kind"],
                "input_resolution_approval",
            )
            action_pause = graph.invoke(
                Command(resume={"action": "approve", "value": "Milwaukee"}),
                config,
            )

        self.assertEqual(action_pause["__interrupt__"][0].value["kind"], "action_approval")
        state = graph.get_state(config).values
        self.assertEqual(state["pending_tool_call"]["name"], "type_text")
        self.assertEqual(state["pending_tool_call"]["arguments"]["text"], "Milwaukee")
        self.assertEqual(state["confirmation"]["input_text"], "Milwaukee")
        self.assertEqual(
            state["input_resolution_history"][0]["original_value"],
            "MKE",
        )
        self.assertIn(
            {"label": "Location", "value": "Milwaukee"},
            state["plan"]["provided_inputs"],
        )

    def test_failed_decision_retries_without_repeating_browser_action(self):
        plan = make_plan()
        snapshot = make_snapshot()
        click_confirmation = make_click_confirmation()
        step_result = StepRunResult(
            planned_step=plan.steps[1],
            confirmation=click_confirmation,
            selected_element=snapshot.observation.elements[0],
            pre_action_snapshot=snapshot,
            post_action_snapshot=snapshot,
            page_changed=True,
            remapped_after_stale=False,
            next_confirmation=None,
        )
        runtime = BrowserRuntime(driver=object(), api_key="test-key")
        config = {"configurable": {"thread_id": "retry-flow"}}

        with (
            patch(
                "src.graphflow.navigate_to_plan",
                return_value=snapshot,
            ),
            patch("src.graphflow.run_next_step", return_value=step_result) as run,
            patch(
                "src.graphflow.confirm_next_step",
                side_effect=[
                    click_confirmation,
                    RuntimeError("temporary decision failure"),
                    make_final_confirmation(),
                ],
            ) as confirm,
        ):
            graph = build_browser_graph(runtime)
            graph.invoke(initial_flow_state(plan), config)
            with self.assertRaisesRegex(RuntimeError, "temporary decision failure"):
                graph.invoke(Command(resume={"action": "approve"}), config)

            self.assertEqual(graph.get_state(config).next, ("decide",))
            final_pause = graph.invoke(None, config)

        self.assertEqual(final_pause["__interrupt__"][0].value["kind"], "final_extraction_approval")
        run.assert_called_once()
        self.assertEqual(confirm.call_count, 3)


if __name__ == "__main__":
    unittest.main()
