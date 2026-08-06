"""Zero-shot planner for the native LIBERO multi-turn Harness loop."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence, Tuple

from embodiedbench.planner.harness.harness_planner import extract_json_object
from embodiedbench.planner.harness.libero_vla_planner import LiberoVLAPlanner


MULTI_TURN_SYSTEM_PROMPT = """You are the planner for a native LIBERO Harness loop.
Emit exactly one JSON primitive per turn and no prose.
The only available primitives are:
{"action":"vla_act","prompt":"<local contact prompt>","target":"<grounded name>","max_chunks":<integer>,"tau":"lift_and_grasp"}
{"action":"move_to","target":"<grounded name>","mode":"above|release_pose","gripper":"close"}
{"action":"release"}
Choose from grounded target names only. Never emit xyz, joint commands, torques,
numeric tolerances, success claims, or multiple primitives. Primitive success
does not imply task success. Use the latest feedback to choose the next action."""


def _valid_cap(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("%s must be an integer greater than zero" % name)


def parse_libero_multi_turn_invocation(
    raw_output: str,
    *,
    available_targets: Sequence[str],
    max_chunks_cap: int,
) -> Optional[Dict[str, object]]:
    """Extract one invocation and fail closed on unknown fields or targets."""
    _valid_cap(max_chunks_cap, "max_chunks_cap")
    targets = {
        target for target in available_targets if isinstance(target, str) and target
    }
    parsed = extract_json_object(raw_output)
    if not isinstance(parsed, dict):
        return None
    action = parsed.get("action")

    if action == "vla_act":
        if set(parsed) != {"action", "prompt", "target", "max_chunks", "tau"}:
            return None
        prompt = parsed.get("prompt")
        target = parsed.get("target")
        max_chunks = parsed.get("max_chunks")
        if not isinstance(prompt, str) or not prompt.strip():
            return None
        if not isinstance(target, str) or target not in targets:
            return None
        if (
            isinstance(max_chunks, bool)
            or not isinstance(max_chunks, int)
            or not 1 <= max_chunks <= max_chunks_cap
        ):
            return None
        if parsed.get("tau") != "lift_and_grasp":
            return None
        return {
            "action": "vla_act",
            "prompt": prompt.strip(),
            "target": target,
            "max_chunks": max_chunks,
            "tau": "lift_and_grasp",
        }

    if action == "move_to":
        if set(parsed) != {"action", "target", "mode", "gripper"}:
            return None
        target = parsed.get("target")
        if not isinstance(target, str) or target not in targets:
            return None
        if parsed.get("mode") not in ("above", "release_pose"):
            return None
        if parsed.get("gripper") != "close":
            return None
        return {
            "action": "move_to",
            "target": target,
            "mode": parsed["mode"],
            "gripper": "close",
        }

    if action == "release":
        return {"action": "release"} if set(parsed) == {"action"} else None
    return None


class LiberoMultiTurnPlanner(LiberoVLAPlanner):
    """Ollama planner that receives semantic state and emits one primitive."""

    def _chat(self, user_prompt: str) -> str:
        from urllib import request

        options = {"temperature": 0}
        if self.think:
            options.update({"num_predict": 8192, "num_ctx": 16384})
        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": MULTI_TURN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "think": self.think,
                "options": options,
            }
        ).encode("utf-8")
        req = request.Request(
            self._endpoint(), data=payload, headers={"Content-Type": "application/json"}
        )
        with request.urlopen(req, timeout=self.request_timeout) as response:
            result = json.load(response)
        message = result.get("message", {})
        self.last_thinking = message.get("thinking") or None
        return message.get("content", "")

    def act_turn(
        self,
        instruction: str,
        state: Dict[str, Any],
        *,
        available_targets: Sequence[str],
        max_chunks_cap: int,
    ) -> Tuple[Optional[Dict[str, object]], str]:
        """Choose one primitive from the current semantic state."""
        _valid_cap(max_chunks_cap, "max_chunks_cap")
        user_prompt = "\n".join(
            [
                "Official task instruction: %s" % instruction,
                "Grounded target names: %s" % json.dumps(list(available_targets)),
                "Remaining vla_act chunk cap: %d" % max_chunks_cap,
                "Current semantic state: %s"
                % json.dumps(state, sort_keys=True, separators=(",", ":")),
                "Return exactly one JSON primitive.",
            ]
        )
        raw_output = self._chat(user_prompt)
        invocation = parse_libero_multi_turn_invocation(
            raw_output,
            available_targets=available_targets,
            max_chunks_cap=max_chunks_cap,
        )
        return invocation, raw_output


__all__ = [
    "LiberoMultiTurnPlanner",
    "MULTI_TURN_SYSTEM_PROMPT",
    "parse_libero_multi_turn_invocation",
]