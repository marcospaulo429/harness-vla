import pytest

from embodiedbench.planner.harness.libero_vla_planner import (
    LiberoVLAPlanner,
    SYSTEM_PROMPT,
    parse_libero_vla_invocation,
)


def test_parser_accepts_tolerant_valid_invocation():
    raw_output = """Result:
```json
{"action":"vla_act","prompt":"pick the bowl","max_chunks":2,"tau":"task_success",}
```
"""

    assert parse_libero_vla_invocation(raw_output, 3) == {
        "action": "vla_act",
        "prompt": "pick the bowl",
        "max_chunks": 2,
        "tau": "task_success",
    }


@pytest.mark.parametrize(
    "raw_output",
    [
        "not JSON",
        '{"action":"grasp","prompt":"x","max_chunks":1,"tau":"task_success"}',
        '{"action":"vla_act","prompt":" ","max_chunks":1,"tau":"task_success"}',
        '{"action":"vla_act","prompt":"x","max_chunks":true,"tau":"task_success"}',
        '{"action":"vla_act","prompt":"x","max_chunks":1,"tau":"object_held"}',
    ],
)
def test_parser_rejects_invalid_invocations(raw_output):
    assert parse_libero_vla_invocation(raw_output, 2) is None


def test_parser_rejects_chunk_count_above_cap():
    raw_output = (
        '{"action":"vla_act","prompt":"pick the bowl",'
        '"max_chunks":3,"tau":"task_success"}'
    )

    assert parse_libero_vla_invocation(raw_output, 2) is None


def test_planner_turn_contains_only_text_task_context():
    class FakePlanner(LiberoVLAPlanner):
        def _chat(self, user_prompt):
            self.seen_prompt = user_prompt
            self.last_thinking = "deliberation"
            return (
                '{"action":"vla_act","prompt":"policy prompt",'
                '"max_chunks":2,"tau":"task_success"}'
            )

    planner = FakePlanner("gemma-test", think=True)
    invocation, raw_output = planner.act("official instruction", 4)

    assert invocation["prompt"] == "policy prompt"
    assert raw_output.startswith('{"action":"vla_act"')
    assert "Official task instruction: official instruction" in planner.seen_prompt
    assert "Available max_chunks cap: 4" in planner.seen_prompt
    assert "no policy action has been executed" in planner.seen_prompt
    assert "pose" not in planner.seen_prompt.lower()
    assert "only one primitive, vla_act" in SYSTEM_PROMPT
    assert "complete Harness primitive library" in SYSTEM_PROMPT