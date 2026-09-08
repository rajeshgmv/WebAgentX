import json
import unittest
from unittest.mock import patch

from selenium.common.exceptions import (
    StaleElementReferenceException,
    WebDriverException,
)

from src.identifysteps import Step, StepPlan
from src.rewriteuserrequest import ProvidedInput
from src.runsteps import (
    CandidateEvaluation,
    ExecutedAction,
    ExtractedDetail,
    ExtractedRecord,
    FinalTaskResult,
    FirstStepRunResult,
    NextStepConfirmation,
    PageElement,
    PageObservation,
    PageSnapshot,
    SecondStepRunResult,
    _observation_prompt_payload,
    collect_page_elements,
    collect_stable_page_elements,
    confirm_next_step,
    extract_final_result,
    run_first_step,
    run_next_step,
    run_second_step,
)


class FakeElement:
    def __init__(
        self,
        tag_name,
        *,
        text="",
        attributes=None,
        accessible_name="",
        displayed=True,
        enabled=True,
        in_popup=False,
        stale_once=False,
        accessible_name_error=False,
        click_error=False,
        inside_open_shadow_root=False,
    ):
        self.tag_name = tag_name
        self.text = text
        self.attributes = attributes or {}
        self._accessible_name = accessible_name
        self.accessible_name_error = accessible_name_error
        self.displayed = displayed
        self.enabled = enabled
        self.in_popup = in_popup
        self.clicked = False
        self.cleared = False
        self.sent_keys = []
        self.stale_once = stale_once
        self.click_error = click_error
        self.inside_open_shadow_root = inside_open_shadow_root

    @property
    def accessible_name(self):
        if self.accessible_name_error:
            raise WebDriverException("Accessible name is unavailable")
        return self._accessible_name

    def get_attribute(self, name):
        return self.attributes.get(name)

    def is_displayed(self):
        if self.stale_once:
            self.stale_once = False
            raise StaleElementReferenceException("Element was replaced")
        return self.displayed

    def is_enabled(self):
        return self.enabled

    def click(self):
        if self.click_error:
            raise WebDriverException("Native click failed")
        self.clicked = True

    def clear(self):
        self.cleared = True

    def send_keys(self, text):
        self.sent_keys.append(text)


class FakeSwitchTo:
    def __init__(self, driver):
        self.driver = driver

    def window(self, handle):
        self.driver.current_window_handle = handle


class FakeDriver:
    def __init__(self, elements=None, shadow_elements=None):
        self.elements = elements or []
        self.shadow_elements = shadow_elements or []
        self.current_url = "https://example.com/landing"
        self.title = "Example landing page"
        self.visited_url = None
        self.window_handles = ["main"]
        self.current_window_handle = "main"
        self.switch_to = FakeSwitchTo(self)

    def get(self, url):
        self.visited_url = url

    @staticmethod
    def _matching_elements(elements, selector):
        if selector == "input":
            return [element for element in elements if element.tag_name == "input"]
        if selector == "input:not([type='hidden'])":
            return [
                element
                for element in elements
                if element.tag_name == "input"
                and element.get_attribute("type") != "hidden"
            ]
        return elements

    def find_elements(self, by, selector):
        return self._matching_elements(self.elements, selector)

    def execute_script(self, script, *arguments):
        if "document.readyState" in script:
            return "complete"
        if "type-text-into-open-shadow-root" in script:
            element, text = arguments
            if not element.inside_open_shadow_root:
                return "not-open-shadow-root"
            element.cleared = True
            element.sent_keys.append(text)
            return "applied"
        if "collect-elements-across-open-shadow-roots" in script:
            return self._matching_elements(
                [*self.elements, *self.shadow_elements],
                arguments[0],
            )
        if "closest" in script:
            return arguments[0].in_popup
        return None


class FakeStructuredLlm:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.response


class SequenceStructuredLlm:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return next(self.responses)


class FakeLlm:
    def __init__(self, response):
        self.structured = FakeStructuredLlm(response)
        self.schema = None
        self.options = None

    def with_structured_output(self, schema, **options):
        self.schema = schema
        self.options = options
        return self.structured


class SequenceFakeLlm(FakeLlm):
    def __init__(self, responses):
        self.structured = SequenceStructuredLlm(responses)
        self.schema = None
        self.options = None


def make_plan():
    return StepPlan(
        url="https://example.com",
        user_request="Search for a product",
        normalized_request=(
            "Locate the requested product and return its available information"
        ),
        ambiguities=[],
        provided_inputs=[
            ProvidedInput(label="product name", value="product")
        ],
        steps=[
            Step(step_number=1, description="Navigate to the homepage"),
            Step(step_number=2, description="Find the product search field"),
            Step(step_number=3, description="Extract the product information"),
        ],
        success_criteria="The product information is extracted",
    )


def make_snapshot(*, index=1, web_element=None):
    element = web_element or FakeElement(
        "input",
        attributes={"type": "search", "placeholder": "Search"},
    )
    return PageSnapshot(
        observation=PageObservation(
            current_url="https://example.com",
            title="Example",
            elements=[
                PageElement(
                    index=index,
                    tag="input",
                    text="",
                    type="search",
                    name="query",
                    id="search",
                    placeholder="Search",
                    aria_label="Search products",
                    accessible_name="Search products",
                    role=None,
                    href=None,
                    enabled=True,
                    in_popup=False,
                )
            ],
        ),
        elements_by_index={index: element},
    )


def make_click_snapshot(
    *,
    index=1,
    web_element=None,
    current_url="https://example.com",
    title="Example",
    text="Open products",
):
    web_element = web_element or FakeElement(
        "a",
        text=text,
        attributes={"href": "https://example.com/products"},
        accessible_name=text,
    )
    return PageSnapshot(
        observation=PageObservation(
            current_url=current_url,
            title=title,
            elements=[
                PageElement(
                    index=index,
                    tag="a",
                    text=text,
                    type=None,
                    name=None,
                    id=None,
                    placeholder=None,
                    aria_label=None,
                    accessible_name=text,
                    role="link",
                    href="https://example.com/products",
                    enabled=True,
                    in_popup=False,
                )
            ],
        ),
        elements_by_index={index: web_element},
    )


def make_click_confirmation():
    return NextStepConfirmation(
        decision="proceed",
        planned_step_number=2,
        confirmed_step_description="Open the product workflow",
        action_type="click",
        target_element_index=1,
        input_text=None,
        step_progress="advance_to_next_step",
        candidate_elements=[
            CandidateEvaluation(
                index=1,
                reason="This link directly opens the relevant workflow.",
            )
        ],
        blocking_element_indices=[],
        expected_result="The product workflow opens.",
        reason="The selected link advances the normalized objective.",
    )


def make_type_confirmation(input_text="John"):
    return NextStepConfirmation(
        decision="proceed",
        planned_step_number=2,
        confirmed_step_description="Enter the product name",
        action_type="type_text",
        target_element_index=1,
        input_text=input_text,
        step_progress="continue_current_step",
        candidate_elements=[
            CandidateEvaluation(
                index=1,
                reason="This is the visible search input for the objective.",
            )
        ],
        blocking_element_indices=[],
        expected_result="Matching autocomplete options should appear.",
        reason="The requested name must be entered before selecting a match.",
    )


def make_final_confirmation():
    return NextStepConfirmation(
        decision="complete",
        planned_step_number=3,
        confirmed_step_description="Extract the product information",
        action_type="none",
        target_element_index=None,
        input_text=None,
        step_progress="continue_current_step",
        candidate_elements=[],
        blocking_element_indices=[],
        expected_result=None,
        reason="The requested product information is visible.",
    )


class RunStepsTests(unittest.TestCase):
    def test_final_result_supports_large_and_grouped_result_sets(self):
        flat_result = FinalTaskResult(
            completed=True,
            summary="Six events were found.",
            details=[
                ExtractedDetail(label=f"Field {index}", value=f"Value {index}")
                for index in range(30)
            ],
            missing_information=[],
        )
        grouped_result = FinalTaskResult(
            completed=True,
            summary="Six events were found.",
            details=[],
            records=[
                ExtractedRecord(
                    title=f"Event {index}",
                    details=[
                        ExtractedDetail(
                            label="Date",
                            value="September 14, 2026",
                        )
                    ],
                )
                for index in range(6)
            ],
            missing_information=[],
        )

        self.assertEqual(len(flat_result.details), 30)
        self.assertEqual(len(grouped_result.records), 6)
        self.assertEqual(
            FinalTaskResult.model_json_schema()["properties"]["details"][
                "maxItems"
            ],
            100,
        )

    def test_blocks_submit_and_requests_missing_required_text(self):
        plan = make_plan()
        required_field = PageElement(
            index=1,
            tag="input",
            text="",
            type="text",
            name="location",
            id=None,
            placeholder="Near (required)",
            aria_label=None,
            accessible_name="Near (required)",
            role=None,
            href=None,
            enabled=True,
            in_popup=False,
            required=True,
            has_value=False,
        )
        submit = PageElement(
            index=2,
            tag="button",
            text="Search",
            type="submit",
            name=None,
            id=None,
            placeholder=None,
            aria_label="Search Providers",
            accessible_name="Search Providers",
            role=None,
            href=None,
            enabled=True,
            in_popup=False,
        )
        snapshot = PageSnapshot(
            observation=PageObservation(
                current_url="https://example.com/search",
                title="Search",
                elements=[required_field, submit],
            ),
            elements_by_index={1: FakeElement("input"), 2: FakeElement("button")},
        )
        proposed_submit = make_click_confirmation().model_copy(
            update={
                "target_element_index": 2,
                "candidate_elements": [
                    CandidateEvaluation(index=2, reason="Submit the search")
                ],
            }
        )
        completed_name_entry = ExecutedAction(
            planned_step_number=2,
            action_type="type_text",
            target_element=make_snapshot().observation.elements[0],
            input_text="product",
        )
        fake_llm = FakeLlm(proposed_submit)

        with patch("src.runsteps.initialize_groq_llm", return_value=fake_llm):
            result = confirm_next_step(
                plan,
                plan.steps[1],
                snapshot,
                api_key="test-key",
                completed_actions=[completed_name_entry],
            )

        self.assertEqual(result.decision, "request_input")
        self.assertEqual(result.action_type, "none")
        self.assertEqual(result.target_element_index, 1)
        self.assertIn("Near (required)", result.requested_input)

    def test_accepts_forward_skip_to_a_later_plan_step(self):
        plan = make_plan()
        snapshot = make_snapshot()
        skip = NextStepConfirmation(
            decision="skip_ahead",
            planned_step_number=2,
            confirmed_step_description=(
                "The current page already satisfies the search step"
            ),
            action_type="none",
            target_element_index=None,
            input_text=None,
            step_progress="advance_to_next_step",
            candidate_elements=[],
            blocking_element_indices=[],
            expected_result=None,
            requested_input=None,
            next_step_number=3,
            reason="The result page directly proves the search already ran.",
        )
        fake_llm = FakeLlm(skip)

        with patch("src.runsteps.initialize_groq_llm", return_value=fake_llm):
            result = confirm_next_step(
                plan,
                plan.steps[1],
                snapshot,
                api_key="test-key",
            )

        self.assertEqual(result.decision, "skip_ahead")
        self.assertEqual(result.next_step_number, 3)
        payload = json.loads(fake_llm.structured.messages[1][1])
        self.assertEqual(
            [step["step_number"] for step in payload["remaining_plan_steps"]],
            [2, 3],
        )

    def test_retries_repeated_text_action_using_completed_action_history(self):
        plan = make_plan()
        snapshot = make_snapshot()
        snapshot.observation.elements[0].has_value = True
        repeated = make_type_confirmation("product")
        corrected = NextStepConfirmation(
            decision="replan",
            planned_step_number=2,
            confirmed_step_description=None,
            action_type="none",
            target_element_index=None,
            input_text=None,
            step_progress="continue_current_step",
            candidate_elements=[],
            blocking_element_indices=[],
            expected_result=None,
            requested_input=None,
            reason="Choose a different action after the completed text entry.",
        )
        completed = ExecutedAction(
            planned_step_number=2,
            action_type="type_text",
            target_element=snapshot.observation.elements[0],
            input_text="product",
        )
        fake_llm = SequenceFakeLlm([repeated, corrected])

        with patch("src.runsteps.initialize_groq_llm", return_value=fake_llm):
            result = confirm_next_step(
                plan,
                plan.steps[1],
                snapshot,
                api_key="test-key",
                completed_actions=[completed],
            )

        self.assertEqual(result.decision, "replan")
        self.assertEqual(len(fake_llm.structured.calls), 2)
        first_payload = json.loads(fake_llm.structured.calls[0][1][1])
        second_payload = json.loads(fake_llm.structured.calls[1][1][1])
        self.assertEqual(len(first_payload["completed_action_history"]), 1)
        self.assertEqual(
            second_payload["rejected_repeated_action"]["action_type"],
            "type_text",
        )

    def test_extracts_grounded_final_result_from_visible_page_text(self):
        final_response = FinalTaskResult(
            completed=True,
            summary="The requested doctor profile was found.",
            details=[
                ExtractedDetail(label="Name", value="Sapna Patel, MD"),
                ExtractedDetail(label="Specialty", value="Family Medicine"),
            ],
            missing_information=["Office hours"],
        )
        fake_llm = FakeLlm(final_response)
        snapshot = make_snapshot()
        snapshot.observation.visible_text = (
            "Sapna Patel, MD\nFamily Medicine\nBurlington, WI 53105"
        )

        with patch(
            "src.runsteps.initialize_groq_llm",
            return_value=fake_llm,
        ):
            result = extract_final_result(
                make_plan(),
                make_plan().steps[-1],
                snapshot,
                api_key="test-key",
            )

        self.assertTrue(result.completed)
        self.assertEqual(result.details[0].value, "Sapna Patel, MD")
        payload = json.loads(fake_llm.structured.messages[1][1])
        self.assertIn("Family Medicine", payload["visible_page_text"])
        self.assertNotIn("elements", payload)

    def test_prompt_budget_prioritizes_required_input_over_extra_links(self):
        elements = [
            PageElement(
                index=index,
                tag="a",
                text=f"Navigation link {index}",
                type=None,
                name=None,
                id=None,
                placeholder=None,
                aria_label=None,
                accessible_name=f"Navigation link {index}",
                role="link",
                href=f"https://example.com/page/{index}",
                enabled=True,
                in_popup=False,
            )
            for index in range(1, 61)
        ]
        elements.append(
            PageElement(
                index=61,
                tag="input",
                text="",
                type="text",
                name=None,
                id=None,
                placeholder="Near (required)",
                aria_label=None,
                accessible_name="Near (required)",
                role=None,
                href=None,
                enabled=True,
                in_popup=False,
                required=True,
                has_value=False,
            )
        )
        observation = PageObservation(
            current_url="https://example.com",
            title="Example",
            elements=elements,
        )

        payload = _observation_prompt_payload(observation)
        included_indices = {element["index"] for element in payload["elements"]}

        self.assertIn(61, included_indices)
        self.assertLessEqual(payload["included_element_count"], 50)
        self.assertGreater(payload["omitted_element_count"], 0)

    def test_collects_text_input_from_an_open_shadow_root(self):
        shadow_input = FakeElement(
            "input",
            attributes={
                "data-testid": "shadow-search-input",
                "type": "text",
                "placeholder": "Search for a provider",
                "role": "combobox",
            },
            accessible_name="Search for a provider",
        )
        driver = FakeDriver(shadow_elements=[shadow_input])

        snapshot = collect_page_elements(driver)

        self.assertEqual(len(snapshot.observation.elements), 1)
        extracted = snapshot.observation.elements[0]
        self.assertEqual(extracted.tag, "input")
        self.assertEqual(extracted.placeholder, "Search for a provider")
        self.assertIs(snapshot.elements_by_index[1], shadow_input)
        diagnostic = snapshot.input_diagnostics[0]
        self.assertEqual(diagnostic.collection_status, "included")
        self.assertEqual(diagnostic.included_index, 1)

    def test_stable_collection_retries_when_dom_changes_mid_capture(self):
        first_snapshot = make_snapshot()
        second_snapshot = make_snapshot()

        with (
            patch("src.runsteps.wait_for_interactive_dom_stable") as wait,
            patch(
                "src.runsteps._useful_element_signature",
                side_effect=["before-first", "after-first", "stable", "stable"],
            ),
            patch(
                "src.runsteps.collect_page_elements",
                side_effect=[first_snapshot, second_snapshot],
            ) as collect,
        ):
            result = collect_stable_page_elements(
                FakeDriver(),
                stability_timeout=6.0,
            )

        self.assertIs(result, second_snapshot)
        self.assertEqual(wait.call_count, 2)
        self.assertEqual(collect.call_count, 2)

    def test_text_input_is_included_and_diagnosed_generically(self):
        driver = FakeDriver(
            [
                FakeElement(
                    "input",
                    attributes={
                        "data-testid": "finder-search-input",
                        "type": "text",
                        "placeholder": (
                            "Specialty, condition, treatment or provider's name"
                        ),
                        "role": "combobox",
                    },
                )
            ]
        )

        snapshot = collect_page_elements(driver)

        self.assertEqual(len(snapshot.observation.elements), 1)
        extracted = snapshot.observation.elements[0]
        self.assertEqual(extracted.tag, "input")
        self.assertEqual(extracted.type, "text")
        self.assertEqual(
            extracted.placeholder,
            "Specialty, condition, treatment or provider's name",
        )
        diagnostic = snapshot.input_diagnostics[0]
        self.assertEqual(diagnostic.data_testid, "finder-search-input")
        self.assertTrue(diagnostic.matched_useful_selector)
        self.assertTrue(diagnostic.displayed)
        self.assertEqual(diagnostic.collection_status, "included")
        self.assertEqual(diagnostic.included_index, 1)

    def test_collects_required_and_nonempty_state_without_collecting_value(self):
        driver = FakeDriver(
            [
                FakeElement(
                    "input",
                    attributes={
                        "type": "text",
                        "placeholder": "Near (required)",
                        "required": "true",
                        "value": "Milwaukee",
                    },
                )
            ]
        )

        snapshot = collect_page_elements(driver)

        extracted = snapshot.observation.elements[0]
        self.assertTrue(extracted.required)
        self.assertTrue(extracted.has_value)
        self.assertNotIn(
            "Milwaukee",
            json.dumps(extracted.model_dump()),
        )

    def test_hidden_input_diagnostic_explains_omission(self):
        driver = FakeDriver(
            [
                FakeElement(
                    "input",
                    attributes={"type": "text", "placeholder": "Search"},
                    displayed=False,
                )
            ]
        )

        snapshot = collect_page_elements(driver)

        self.assertEqual(snapshot.observation.elements, [])
        diagnostic = snapshot.input_diagnostics[0]
        self.assertFalse(diagnostic.displayed)
        self.assertEqual(diagnostic.collection_status, "hidden")

    def test_input_property_error_is_visible_in_diagnostics(self):
        driver = FakeDriver(
            [
                FakeElement(
                    "input",
                    attributes={"type": "text", "placeholder": "Search"},
                    accessible_name_error=True,
                )
            ]
        )

        snapshot = collect_page_elements(driver)

        self.assertEqual(len(snapshot.observation.elements), 1)
        diagnostic = snapshot.input_diagnostics[0]
        self.assertEqual(diagnostic.collection_status, "included")
        self.assertIn("WebDriverException", diagnostic.collection_error)
        self.assertTrue(
            any("accessible_name" in error for error in diagnostic.diagnostic_errors)
        )

    def test_collects_only_visible_useful_elements(self):
        driver = FakeDriver(
            [
                FakeElement(
                    "button",
                    text="  Accept   cookies ",
                    attributes={
                        "aria-label": "Accept cookies",
                        "href": "https://example.com/accept?session=secret#dialog",
                    },
                    in_popup=True,
                ),
                FakeElement("button", text="Hidden", displayed=False),
                FakeElement("button"),
            ]
        )

        snapshot = collect_page_elements(driver)

        self.assertEqual(len(snapshot.observation.elements), 1)
        element = snapshot.observation.elements[0]
        self.assertEqual(element.index, 1)
        self.assertEqual(element.text, "Accept cookies")
        self.assertEqual(element.href, "https://example.com/accept")
        self.assertTrue(element.in_popup)
        self.assertIs(snapshot.elements_by_index[1], driver.elements[0])

    def test_confirms_next_step_with_strict_structured_output(self):
        response = NextStepConfirmation(
            decision="proceed",
            planned_step_number=2,
            confirmed_step_description="Find the product search field",
            action_type="click",
            target_element_index=1,
            input_text=None,
            step_progress="advance_to_next_step",
            candidate_elements=[
                CandidateEvaluation(
                    index=1,
                    reason="The field directly supports the complete objective.",
                )
            ],
            blocking_element_indices=[],
            expected_result="A product search can be entered.",
            reason="The search field is visible.",
        )
        fake_llm = FakeLlm(response)
        snapshot = make_snapshot()

        with patch(
            "src.runsteps.initialize_groq_llm",
            return_value=fake_llm,
        ):
            result = confirm_next_step(
                make_plan(),
                Step(step_number=2, description="Find the product search field"),
                snapshot,
                api_key="test-key",
            )

        self.assertEqual(result.decision, "proceed")
        self.assertEqual(fake_llm.schema, NextStepConfirmation)
        self.assertEqual(
            fake_llm.options,
            {"method": "function_calling", "include_raw": True},
        )
        self.assertEqual(fake_llm.structured.messages[0][0], "system")
        self.assertEqual(fake_llm.structured.messages[1][0], "human")
        payload = json.loads(fake_llm.structured.messages[1][1])
        self.assertNotIn("user_request", payload)
        self.assertNotIn("input_diagnostics", payload["page_observation"])
        self.assertEqual(
            payload["normalized_objective"],
            "Locate the requested product and return its available information",
        )
        self.assertEqual(payload["known_ambiguities"], [])
        self.assertEqual(
            payload["provided_inputs"],
            [{"label": "product name", "value": "product"}],
        )
        self.assertEqual(
            payload["success_criteria"],
            "The product information is extracted",
        )

    def test_confirms_grounded_type_text_action(self):
        response = make_type_confirmation("product")
        fake_llm = FakeLlm(response)

        with patch(
            "src.runsteps.initialize_groq_llm",
            return_value=fake_llm,
        ):
            result = confirm_next_step(
                make_plan(),
                Step(step_number=2, description="Find the product search field"),
                make_snapshot(),
                api_key="test-key",
            )

        self.assertEqual(result.action_type, "type_text")
        self.assertEqual(result.input_text, "product")
        self.assertEqual(result.step_progress, "continue_current_step")

    def test_confirmation_receives_the_last_completed_action(self):
        response = make_click_confirmation()
        fake_llm = FakeLlm(response)
        last_action = ExecutedAction(
            planned_step_number=2,
            action_type="type_text",
            target_element=make_snapshot().observation.elements[0],
            input_text="product",
        )

        with patch(
            "src.runsteps.initialize_groq_llm",
            return_value=fake_llm,
        ):
            confirm_next_step(
                make_plan(),
                Step(step_number=2, description="Find the product search field"),
                make_click_snapshot(),
                api_key="test-key",
                last_executed_action=last_action,
            )

        payload = json.loads(fake_llm.structured.messages[1][1])
        self.assertEqual(
            payload["last_executed_action"]["action_type"],
            "type_text",
        )
        self.assertEqual(
            payload["last_executed_action"]["input_text"],
            "product",
        )

    def test_accepts_request_for_missing_required_safe_text(self):
        required_element = FakeElement(
            "input",
            attributes={
                "type": "text",
                "placeholder": "Near (required)",
                "required": "true",
                "value": "",
            },
        )
        snapshot = collect_page_elements(FakeDriver([required_element]))
        response = NextStepConfirmation(
            decision="request_input",
            planned_step_number=2,
            confirmed_step_description="Supply the required location",
            action_type="none",
            target_element_index=1,
            input_text=None,
            step_progress="continue_current_step",
            candidate_elements=[
                CandidateEvaluation(
                    index=1,
                    reason="The required location field is empty.",
                )
            ],
            blocking_element_indices=[],
            expected_result=None,
            requested_input="What location should be used?",
            reason="A location is required before submitting the search.",
        )
        fake_llm = FakeLlm(response)

        with patch(
            "src.runsteps.initialize_groq_llm",
            return_value=fake_llm,
        ):
            result = confirm_next_step(
                make_plan(),
                Step(step_number=2, description="Find the product search field"),
                snapshot,
                api_key="test-key",
            )

        self.assertEqual(result.decision, "request_input")
        self.assertEqual(result.target_element_index, 1)

    def test_allows_actionable_confirmation_without_expected_result(self):
        response = make_click_confirmation().model_copy(
            update={"expected_result": None}
        )
        fake_llm = FakeLlm(response)

        with patch(
            "src.runsteps.initialize_groq_llm",
            return_value=fake_llm,
        ):
            result = confirm_next_step(
                make_plan(),
                Step(step_number=2, description="Find the product search field"),
                make_click_snapshot(),
                api_key="test-key",
            )

        self.assertIsNone(result.expected_result)

    def test_retries_non_action_with_target_as_a_consistency_error(self):
        rejected = make_click_confirmation().model_copy(
            update={
                "action_type": "none",
                "step_progress": "continue_current_step",
            }
        )
        corrected = make_click_confirmation()
        fake_llm = SequenceFakeLlm([rejected, corrected])

        with patch(
            "src.runsteps.initialize_groq_llm",
            return_value=fake_llm,
        ):
            result = confirm_next_step(
                make_plan(),
                make_plan().steps[1],
                make_click_snapshot(),
                api_key="test-key",
            )

        self.assertEqual(result.action_type, "click")
        self.assertEqual(result.target_element_index, 1)
        self.assertEqual(len(fake_llm.structured.calls), 2)
        corrected_payload = json.loads(fake_llm.structured.calls[1][1][1])
        self.assertIn("rejected_confirmation", corrected_payload)
        self.assertIn(
            "cannot include a target element",
            corrected_payload["rejected_confirmation"]["validation_error"],
        )

    def test_retries_proceed_without_action_or_progress(self):
        rejected = make_click_confirmation().model_copy(
            update={
                "action_type": "none",
                "target_element_index": None,
                "step_progress": "continue_current_step",
                "candidate_elements": [],
            }
        )
        corrected = make_click_confirmation()
        fake_llm = SequenceFakeLlm([rejected, corrected])

        with patch(
            "src.runsteps.initialize_groq_llm",
            return_value=fake_llm,
        ):
            result = confirm_next_step(
                make_plan(),
                make_plan().steps[1],
                make_click_snapshot(),
                api_key="test-key",
            )

        self.assertEqual(result.action_type, "click")
        self.assertEqual(len(fake_llm.structured.calls), 2)
        corrected_payload = json.loads(fake_llm.structured.calls[1][1][1])
        self.assertIn(
            "must advance a satisfied step",
            corrected_payload["rejected_confirmation"]["validation_error"],
        )

    def test_rejects_input_text_attached_to_click_action(self):
        response = make_click_confirmation().model_copy(
            update={"input_text": "unexpected text"}
        )
        fake_llm = FakeLlm(response)

        with patch(
            "src.runsteps.initialize_groq_llm",
            return_value=fake_llm,
        ):
            with self.assertRaisesRegex(ValueError, "Only a type_text"):
                confirm_next_step(
                    make_plan(),
                    Step(
                        step_number=2,
                        description="Find the product search field",
                    ),
                    make_click_snapshot(),
                    api_key="test-key",
                )

    def test_normalizes_type_text_to_same_step_verification(self):
        response = make_type_confirmation("product").model_copy(
            update={"step_progress": "advance_to_next_step"}
        )
        fake_llm = FakeLlm(response)

        with patch(
            "src.runsteps.initialize_groq_llm",
            return_value=fake_llm,
        ):
            result = confirm_next_step(
                make_plan(),
                Step(
                    step_number=2,
                    description="Find the product search field",
                ),
                make_snapshot(),
                api_key="test-key",
            )

        self.assertEqual(result.action_type, "type_text")
        self.assertEqual(result.step_progress, "continue_current_step")

    def test_normalizes_non_action_to_same_step(self):
        response = make_final_confirmation().model_copy(
            update={"step_progress": "advance_to_next_step"}
        )
        fake_llm = FakeLlm(response)

        with patch(
            "src.runsteps.initialize_groq_llm",
            return_value=fake_llm,
        ):
            result = confirm_next_step(
                make_plan(),
                make_plan().steps[-1],
                make_snapshot(),
                api_key="test-key",
            )

        self.assertEqual(result.action_type, "none")
        self.assertEqual(result.step_progress, "continue_current_step")

    def test_allows_proceed_non_action_to_advance_a_satisfied_step(self):
        response = make_final_confirmation().model_copy(
            update={
                "decision": "proceed",
                "step_progress": "advance_to_next_step",
                "reason": "The current page already satisfies this plan step.",
            }
        )
        fake_llm = FakeLlm(response)

        with patch(
            "src.runsteps.initialize_groq_llm",
            return_value=fake_llm,
        ):
            result = confirm_next_step(
                make_plan(),
                make_plan().steps[-1],
                make_snapshot(),
                api_key="test-key",
            )

        self.assertEqual(result.action_type, "none")
        self.assertEqual(result.step_progress, "advance_to_next_step")

    def test_rejects_unknown_element_index_from_llm(self):
        response = NextStepConfirmation(
            decision="proceed",
            planned_step_number=2,
            confirmed_step_description="Find the product search field",
            action_type="click",
            target_element_index=99,
            input_text=None,
            step_progress="advance_to_next_step",
            candidate_elements=[
                CandidateEvaluation(
                    index=99,
                    reason="This index was not present in the observation.",
                )
            ],
            blocking_element_indices=[],
            expected_result="A product search can be entered.",
            reason="Use an element that does not exist.",
        )
        fake_llm = FakeLlm(response)

        with patch(
            "src.runsteps.initialize_groq_llm",
            return_value=fake_llm,
        ):
            with self.assertRaisesRegex(ValueError, "unknown element index"):
                confirm_next_step(
                    make_plan(),
                    Step(
                        step_number=2,
                        description="Find the product search field",
                    ),
                    make_snapshot(),
                    api_key="test-key",
                )

    def test_rejects_target_missing_from_candidate_list(self):
        response = NextStepConfirmation(
            decision="proceed",
            planned_step_number=2,
            confirmed_step_description="Find the product search field",
            action_type="click",
            target_element_index=1,
            input_text=None,
            step_progress="advance_to_next_step",
            candidate_elements=[],
            blocking_element_indices=[],
            expected_result="A product search can be entered.",
            reason="The search field is visible.",
        )
        fake_llm = FakeLlm(response)

        with patch(
            "src.runsteps.initialize_groq_llm",
            return_value=fake_llm,
        ):
            with self.assertRaisesRegex(ValueError, "candidate list"):
                confirm_next_step(
                    make_plan(),
                    Step(
                        step_number=2,
                        description="Find the product search field",
                    ),
                    make_snapshot(),
                    api_key="test-key",
                )

    def test_first_step_navigates_and_reviews_second_step(self):
        driver = FakeDriver()
        snapshot = make_snapshot()
        confirmation = NextStepConfirmation(
            decision="proceed",
            planned_step_number=2,
            confirmed_step_description="Find the product search field",
            action_type="click",
            target_element_index=1,
            input_text=None,
            step_progress="advance_to_next_step",
            candidate_elements=[
                CandidateEvaluation(
                    index=1,
                    reason="The field directly supports the complete objective.",
                )
            ],
            blocking_element_indices=[],
            expected_result="A product search can be entered.",
            reason="The search field is visible.",
        )

        with (
            patch("src.runsteps.wait_for_page_ready") as wait,
            patch("src.runsteps.collect_stable_page_elements", return_value=snapshot),
            patch("src.runsteps.confirm_next_step", return_value=confirmation) as confirm,
        ):
            result = run_first_step(
                driver,
                make_plan(),
                api_key="test-key",
                settle_seconds=0,
            )

        self.assertEqual(driver.visited_url, "https://example.com")
        wait.assert_called_once_with(driver, timeout=20.0)
        confirm.assert_called_once()
        self.assertEqual(result.completed_step.step_number, 1)
        self.assertEqual(result.confirmation.decision, "proceed")

    def test_second_step_executes_one_click_and_collects_fresh_snapshot(self):
        plan = make_plan()
        pre_action = make_click_snapshot()
        post_action = make_click_snapshot(
            current_url="https://example.com/products",
            title="Products",
            text="Browse products",
        )
        first_result = FirstStepRunResult(
            completed_step=plan.steps[0],
            snapshot=pre_action,
            confirmation=make_click_confirmation(),
        )
        driver = FakeDriver()

        with (
            patch("src.runsteps.wait_for_page_ready") as wait,
            patch(
                "src.runsteps.collect_stable_page_elements",
                return_value=post_action,
            ) as collect,
            patch(
                "src.runsteps.confirm_next_step",
                return_value=make_final_confirmation(),
            ) as confirm,
        ):
            result = run_second_step(
                driver,
                plan,
                first_result,
                api_key="test-key",
                settle_seconds=0,
            )

        self.assertTrue(pre_action.elements_by_index[1].clicked)
        wait.assert_called_once_with(driver, timeout=20.0)
        collect.assert_called_once_with(driver, max_elements=80)
        confirm.assert_called_once_with(
            plan,
            plan.steps[2],
            post_action,
            api_key="test-key",
            last_executed_action=result.executed_action,
        )
        self.assertTrue(result.page_changed)
        self.assertFalse(result.remapped_after_stale)
        self.assertIs(result.post_action_snapshot, post_action)
        self.assertEqual(result.next_confirmation.planned_step_number, 3)

    def test_type_text_action_enters_text_and_reviews_same_step(self):
        plan = make_plan()
        pre_action = make_snapshot()
        post_action = make_snapshot()
        next_confirmation = make_click_confirmation()
        driver = FakeDriver()

        with (
            patch("src.runsteps.wait_for_page_ready") as wait,
            patch(
                "src.runsteps.collect_stable_page_elements",
                return_value=post_action,
            ),
            patch(
                "src.runsteps.confirm_next_step",
                return_value=next_confirmation,
            ) as confirm,
        ):
            result = run_next_step(
                driver,
                plan,
                plan.steps[1],
                pre_action,
                make_type_confirmation("John"),
                api_key="test-key",
                settle_seconds=0,
            )

        input_element = pre_action.elements_by_index[1]
        self.assertTrue(input_element.clicked)
        self.assertTrue(input_element.cleared)
        self.assertEqual(input_element.sent_keys, ["John"])
        wait.assert_called_once_with(driver, timeout=20.0)
        confirm.assert_called_once_with(
            plan,
            plan.steps[1],
            post_action,
            api_key="test-key",
            last_executed_action=result.executed_action,
        )
        self.assertTrue(result.page_changed)
        self.assertEqual(result.next_confirmation.planned_step_number, 2)

    def test_type_text_uses_verified_fallback_for_open_shadow_input(self):
        plan = make_plan()
        shadow_input = FakeElement(
            "input",
            attributes={"type": "search", "placeholder": "Search"},
            click_error=True,
            inside_open_shadow_root=True,
        )
        pre_action = make_snapshot(web_element=shadow_input)
        post_action = make_snapshot()

        with (
            patch("src.runsteps.wait_for_page_ready"),
            patch(
                "src.runsteps.collect_stable_page_elements",
                return_value=post_action,
            ),
            patch(
                "src.runsteps.confirm_next_step",
                return_value=make_click_confirmation(),
            ),
        ):
            result = run_next_step(
                FakeDriver(),
                plan,
                plan.steps[1],
                pre_action,
                make_type_confirmation("John"),
                api_key="test-key",
                settle_seconds=0,
            )

        self.assertTrue(shadow_input.cleared)
        self.assertEqual(shadow_input.sent_keys, ["John"])
        self.assertTrue(result.page_changed)

    def test_type_text_does_not_use_dom_fallback_for_light_dom_input(self):
        plan = make_plan()
        light_dom_input = FakeElement(
            "input",
            attributes={"type": "search", "placeholder": "Search"},
            click_error=True,
            inside_open_shadow_root=False,
        )

        with self.assertRaisesRegex(WebDriverException, "Native click failed"):
            run_next_step(
                FakeDriver(),
                plan,
                plan.steps[1],
                make_snapshot(web_element=light_dom_input),
                make_type_confirmation("John"),
                api_key="test-key",
                settle_seconds=0,
            )

        self.assertFalse(light_dom_input.cleared)
        self.assertEqual(light_dom_input.sent_keys, [])

    def test_type_text_rejects_password_inputs(self):
        plan = make_plan()
        password_element = FakeElement(
            "input",
            attributes={"type": "password"},
        )
        password_snapshot = PageSnapshot(
            observation=PageObservation(
                current_url="https://example.com",
                title="Example",
                elements=[
                    PageElement(
                        index=1,
                        tag="input",
                        text="",
                        type="password",
                        name="password",
                        id="password",
                        placeholder="Password",
                        aria_label="Password",
                        accessible_name="Password",
                        role=None,
                        href=None,
                        enabled=True,
                        in_popup=False,
                    )
                ],
            ),
            elements_by_index={1: password_element},
        )

        with self.assertRaisesRegex(ValueError, "supported input or textarea"):
            run_next_step(
                FakeDriver(),
                plan,
                plan.steps[1],
                password_snapshot,
                make_type_confirmation("not-a-real-password"),
                api_key="test-key",
                settle_seconds=0,
            )

        self.assertEqual(password_element.sent_keys, [])

    def test_type_text_remaps_same_input_after_stale_reference(self):
        plan = make_plan()
        stale_input = FakeElement(
            "input",
            attributes={"type": "search", "placeholder": "Search"},
            stale_once=True,
        )
        fresh_input = FakeElement(
            "input",
            attributes={"type": "search", "placeholder": "Search"},
        )
        pre_action = make_snapshot(web_element=stale_input)
        refreshed = make_snapshot(index=7, web_element=fresh_input)
        post_action = make_snapshot()

        with (
            patch("src.runsteps.wait_for_page_ready"),
            patch(
                "src.runsteps.collect_stable_page_elements",
                side_effect=[refreshed, post_action],
            ),
            patch(
                "src.runsteps.confirm_next_step",
                return_value=make_click_confirmation(),
            ) as confirm,
        ):
            result = run_next_step(
                FakeDriver(),
                plan,
                plan.steps[1],
                pre_action,
                make_type_confirmation("John"),
                api_key="test-key",
                settle_seconds=0,
            )

        self.assertEqual(fresh_input.sent_keys, ["John"])
        self.assertTrue(result.remapped_after_stale)
        self.assertEqual(result.selected_element.index, 7)
        confirm.assert_called_once_with(
            plan,
            plan.steps[1],
            post_action,
            api_key="test-key",
            last_executed_action=result.executed_action,
        )

    def test_second_step_remaps_same_element_after_stale_reference(self):
        plan = make_plan()
        stale_element = FakeElement(
            "a",
            text="Open products",
            attributes={"href": "https://example.com/products"},
            accessible_name="Open products",
            stale_once=True,
        )
        fresh_element = FakeElement(
            "a",
            text="Open products",
            attributes={"href": "https://example.com/products"},
            accessible_name="Open products",
        )
        pre_action = make_click_snapshot(web_element=stale_element)
        refreshed = make_click_snapshot(index=7, web_element=fresh_element)
        post_action = make_click_snapshot(
            index=1,
            current_url="https://example.com/products",
            title="Products",
            text="Browse products",
        )
        first_result = FirstStepRunResult(
            completed_step=plan.steps[0],
            snapshot=pre_action,
            confirmation=make_click_confirmation(),
        )
        driver = FakeDriver()

        with (
            patch("src.runsteps.wait_for_page_ready"),
            patch(
                "src.runsteps.collect_stable_page_elements",
                side_effect=[refreshed, post_action],
            ),
            patch(
                "src.runsteps.confirm_next_step",
                return_value=make_final_confirmation(),
            ) as confirm,
        ):
            result = run_second_step(
                driver,
                plan,
                first_result,
                api_key="test-key",
                settle_seconds=0,
            )

        self.assertTrue(fresh_element.clicked)
        confirm.assert_called_once_with(
            plan,
            plan.steps[2],
            post_action,
            api_key="test-key",
            last_executed_action=result.executed_action,
        )
        self.assertTrue(result.remapped_after_stale)
        self.assertEqual(result.selected_element.index, 7)

    def test_generic_runner_refuses_to_execute_the_final_plan_step(self):
        plan = make_plan()

        with self.assertRaisesRegex(ValueError, "final result step"):
            run_next_step(
                FakeDriver(),
                plan,
                plan.steps[-1],
                make_click_snapshot(),
                make_final_confirmation(),
                api_key="test-key",
                settle_seconds=0,
            )

    def test_popup_action_reviews_the_same_plan_step_again(self):
        plan = make_plan()
        popup_confirmation = make_click_confirmation().model_copy(
            update={
                "decision": "handle_popup",
                "step_progress": "continue_current_step",
            }
        )
        pre_action = make_click_snapshot()
        post_action = make_click_snapshot(text="Find products")

        with (
            patch("src.runsteps.wait_for_page_ready"),
            patch(
                "src.runsteps.collect_stable_page_elements",
                return_value=post_action,
            ),
            patch(
                "src.runsteps.confirm_next_step",
                return_value=make_click_confirmation(),
            ) as confirm,
        ):
            result = run_next_step(
                FakeDriver(),
                plan,
                plan.steps[1],
                pre_action,
                popup_confirmation,
                api_key="test-key",
                settle_seconds=0,
            )

        confirm.assert_called_once_with(
            plan,
            plan.steps[1],
            post_action,
            api_key="test-key",
            last_executed_action=result.executed_action,
        )
        self.assertEqual(result.next_confirmation.planned_step_number, 2)


if __name__ == "__main__":
    unittest.main()
