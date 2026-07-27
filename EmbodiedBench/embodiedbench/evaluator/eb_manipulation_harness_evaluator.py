"""Evaluator for the Harness VLA beta on EB-Manipulation.

Mirrors :class:`EB_ManipulationEvaluator` but drives the environment with the
Harness architecture: the :class:`HarnessPlanner` emits ONE primitive invocation
per turn, the :class:`PrimitiveLibrary` compiles it into a short burst of discrete
actions, and the loop closes by re-perceiving object coordinates and the
end-effector pose before the next turn. A per-episode JSONL audit trace records
every primitive, its compiled actions, and the environment feedback.

Beta scope (see ``docs/HARNESS_VLA_NOT_IMPLEMENTED.md``):
* language-only perception (object coordinate table as text, no images);
* zero-shot (fixed manual Global Memory seed, no Task Specific Memory);
* ``vla_act`` is a mock scripted contact primitive, not a frozen VLA.
"""

import os
import copy
import json

import numpy as np
from tqdm import tqdm

from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv, ValidEvalSets
from embodiedbench.envs.eb_manipulation.eb_man_utils import form_harness_grounding_for_input
from embodiedbench.planner.harness.global_memory import GlobalMemory
from embodiedbench.planner.harness.harness_planner import HarnessPlanner
from embodiedbench.planner.harness.evaluation_guards import (
    NoProgressGuard,
    validate_vla_semantics,
)
from embodiedbench.planner.harness.primitives import (
    PoseState,
    PrimitiveError,
    PrimitiveLibrary,
    classify_grasp_outcome,
    pose_from_observation,
    primitive_termination,
)
from embodiedbench.main import logger


class EB_ManipulationHarnessEvaluator:
    """Harness VLA beta evaluator (one primitive per planner turn)."""

    def __init__(self, config):
        self.model_name = config['model_name']
        self.config = config
        self.eval_set = ValidEvalSets[0]
        self.env = None
        self.planner = None
        self.library = PrimitiveLibrary(
            approach_dz=config.get('approach_dz', 8),
            lift_dz=config.get('lift_dz', 6),
        )
        # Turns (primitive invocations), not env steps, cap per episode.
        self.max_turns = config.get('max_turns', 12)
        # vla_act consumes several env steps, so allow more than the default 15.
        self.max_env_steps = config.get('max_env_steps', 30)
        self.grasp_thresholds = {
            'object_lift_threshold': config.get('grasp_object_lift_threshold', 3.0),
            'max_gripper_object_distance': config.get('grasp_max_distance', 8.0),
            'empty_object_motion_threshold': config.get('grasp_empty_motion_threshold', 1.0),
            'min_gripper_lift': config.get('grasp_min_gripper_lift', 3.0),
            'max_comotion_residual': config.get('grasp_max_comotion_residual', 2.0),
        }

    # -- persistence ------------------------------------------------------

    def _results_dir(self):
        res_path = os.path.join(self.env.log_path, 'results')
        os.makedirs(res_path, exist_ok=True)
        return res_path

    def save_episode_metric(self, episode_info):
        filename = 'episode_{}_res.json'.format(self.env._current_episode_num)
        with open(os.path.join(self._results_dir(), filename), 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False)

    def save_trace(self, trace):
        filename = 'trace_episode_{}.jsonl'.format(self.env._current_episode_num)
        with open(os.path.join(self._results_dir(), filename), 'w', encoding='utf-8') as f:
            for record in trace:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def print_task_eval_results(self, filename):
        folder_path = self._results_dir()
        total, success, planner_steps, output_error = 0, 0, 0, 0
        semantic_rejects, no_progress_rejected = 0, 0
        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith(".json") and file_name.startswith("episode"):
                with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                if data.get("planner_output_error", 0) > 0:
                    output_error += 1
                if data.get("task_success") == 1:
                    success += 1
                planner_steps += data.get("planner_steps", 0)
                semantic_rejects += data.get("semantic_rejects", 0)
                no_progress_rejected += data.get("no_progress_rejected", 0)
                total += 1
        task_log = {
            'save_path': self.log_path,
            'total_num_tasks': total,
            'num_success': success,
            'success_rate': success / total if total else 0.0,
            'avg_planner_steps': planner_steps / total if total else 0.0,
            'output_format_error': output_error,
            'semantic_rejects': semantic_rejects,
            'no_progress_rejected': no_progress_rejected,
        }
        with open(os.path.join(folder_path, filename), 'w', encoding='utf-8') as f:
            json.dump(task_log, f, ensure_ascii=False)

    # -- perception helpers ----------------------------------------------

    def _perceive_grounding(self, obs):
        coords, roles, labels, id_to_sim_name = form_harness_grounding_for_input(
            copy.deepcopy(obs), self.env.task_class, ['front_rgb']
        )
        coords = {k: [int(round(v)) for v in coord] for k, coord in coords.items()}
        return coords, roles, labels, id_to_sim_name

    def _planner_act(self, instruction, coords, pose, history, roles, labels):
        try:
            return self.planner.act(
                instruction, coords, pose, history,
                object_roles=roles, object_labels=labels,
            )
        except TypeError as exc:
            if 'unexpected keyword argument' not in str(exc):
                raise
            return self.planner.act(instruction, coords, pose, history)

    def _current_pose(self, obs):
        pose = pose_from_observation(obs)
        return pose if pose is not None else PoseState()

    def _grasp_attachment_evidence(self, info=None):
        evidence_getter = getattr(self.env, 'get_grasp_attachment_evidence', None)
        if callable(evidence_getter):
            try:
                names, available = evidence_getter()
                return list(names), bool(available)
            except Exception:
                pass
        getter = getattr(self.env, 'get_grasped_object_names', None)
        if callable(getter):
            try:
                return list(getter()), True
            except Exception:
                pass
        if 'grasped_objects' in (info or {}):
            return list(info.get('grasped_objects', []) or []), True
        return [], False

    def _grasped_object_names(self, info=None):
        return self._grasp_attachment_evidence(info)[0]

    # -- main loop --------------------------------------------------------

    def evaluate(self):
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {
                'reward': [], 'action_success': [],
                'semantic_rejects': 0, 'no_progress_rejected': 0,
            }
            trace = []

            _, obs = self.env.reset()
            obs_dict = vars(copy.deepcopy(obs))
            if self.config.get('save_images', False):
                self.env.save_image(['front_rgb'])
            avg_obj_coord, object_roles, object_labels, id_to_sim_name = (
                self._perceive_grounding(obs_dict)
            )
            pose = self._current_pose(obs_dict)
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")

            self.planner.reset()
            history = []
            held_object_id = None
            no_progress_guard = NoProgressGuard(limit=3)
            done = False
            info = {'task_success': 0, 'episode_elapsed_seconds': 0}
            turn = 0

            while not done and turn < self.max_turns:
                turn += 1
                invocation, raw_text = self._planner_act(
                    user_instruction, avg_obj_coord, pose.as_action(), history,
                    object_roles, object_labels,
                )
                record = {
                    'turn': turn,
                    'pose_before': pose.as_action(),
                    'object_coords': avg_obj_coord,
                    'object_roles': object_roles,
                    'object_labels': object_labels,
                    'id_to_sim_name': id_to_sim_name,
                    'raw_output': raw_text,
                    'invocation': invocation,
                }

                if invocation is None:
                    feedback = "Output was not valid JSON with an 'action'. Reply with one primitive as JSON."
                    record['status'] = 'parse_error'
                    record['feedback'] = feedback
                    history.append({'action': None, 'status': 'parse_error', 'feedback': feedback})
                    trace.append(record)
                    continue

                semantic_rejection = validate_vla_semantics(
                    invocation, avg_obj_coord, object_roles, held_object_id, object_labels
                )
                if semantic_rejection is not None:
                    status, feedback = semantic_rejection
                    record.update({
                        'execution_status': 'not_executed',
                        'status': status,
                        'semantic_reject': True,
                        'feedback': feedback,
                        'held_object_id': held_object_id,
                    })
                    episode_info['semantic_rejects'] += 1
                    history.append({'action': invocation, 'status': status, 'feedback': feedback})
                    trace.append(record)
                    continue

                if no_progress_guard.should_reject(invocation):
                    feedback = (
                        "Repeated action rejected before execution after 3 identical zero-reward, "
                        "zero-task-progress executions. Choose a different action or re-stage; the "
                        "episode remains active."
                    )
                    record.update({
                        'execution_status': 'not_executed',
                        'status': 'no_progress_rejected',
                        'no_progress_rejected': True,
                        'feedback': feedback,
                        'held_object_id': held_object_id,
                    })
                    episode_info['no_progress_rejected'] += 1
                    history.append({
                        'action': invocation, 'status': 'no_progress_rejected', 'feedback': feedback
                    })
                    trace.append(record)
                    continue

                # Compile the primitive to discrete actions.
                try:
                    result = self.library.compile(invocation, pose, avg_obj_coord)
                except PrimitiveError as e:
                    feedback = f"Primitive could not be executed: {e}"
                    record['status'] = 'compile_error'
                    record['feedback'] = feedback
                    history.append({'action': invocation, 'status': 'compile_error', 'feedback': feedback})
                    trace.append(record)
                    continue

                mode = result.meta.get('mode')

                # Execute the compiled actions (bounded by remaining env steps).
                remaining = self.env._max_episode_steps - self.env._current_step
                step_results = []
                last_feedback = ''
                object_id = result.meta.get('object_id')
                position_pre_grasp = avg_obj_coord.get(object_id) if mode == 'grasp' else None
                gripper_pre_grasp = pose.as_action()[:3] if mode == 'grasp' else None
                position_at_close = None
                gripper_at_close = None
                attachment_at_close = []
                attachment_evidence_available = False
                for subaction_index, action_single in enumerate(result.actions[:max(0, remaining)]):
                    obs, reward, done, info = self.env.step(action_single)
                    if self.config.get('save_images', False):
                        self.env.save_image(['front_rgb'])
                    last_feedback = info.get('env_feedback', '')
                    sub_obs_dict = vars(copy.deepcopy(obs)) if not isinstance(obs, dict) else obs
                    sub_coords, _, _, _ = self._perceive_grounding(sub_obs_dict)
                    sub_pose = self._current_pose(sub_obs_dict)
                    attachments, attachment_available = self._grasp_attachment_evidence(info)
                    attachment_evidence_available = (
                        attachment_evidence_available or attachment_available
                    )
                    step_results.append({
                        'action': list(action_single),
                        'reward': reward,
                        'action_success': info['action_success'],
                        'task_success': info['task_success'],
                        'env_done': bool(done),
                        'env_feedback': last_feedback,
                        'grasped_objects': attachments,
                        'attachment_evidence_available': attachment_available,
                    })
                    if mode == 'grasp' and subaction_index == 2:
                        position_at_close = sub_coords.get(object_id)
                        gripper_at_close = sub_pose.as_action()[:3]
                        attachment_at_close = attachments
                    episode_info['reward'].append(reward)
                    episode_info['action_success'].append(info['action_success'])
                    print(f"Executed {result.name} sub-action {action_single}, task_success={info['task_success']}")
                    if done:
                        break

                obs_dict = vars(copy.deepcopy(obs)) if not isinstance(obs, dict) else obs
                avg_obj_coord, object_roles, object_labels, id_to_sim_name = (
                    self._perceive_grounding(obs_dict)
                )
                pose = self._current_pose(obs_dict)

                execution_status = 'success' if step_results and step_results[-1]['action_success'] == 1.0 else 'failed'
                feedback = last_feedback or 'no environment feedback'
                grasp_outcome = None
                release_executed = bool(
                    mode == 'place'
                    and any(step['action'][6] == 1 for step in step_results)
                )
                if mode == 'grasp':
                    final_attachments = (
                        step_results[-1]['grasped_objects'] if step_results else attachment_at_close
                    )
                    grasp_outcome = classify_grasp_outcome(
                        target_object_id=object_id,
                        target_sim_name=id_to_sim_name.get(object_id),
                        grasped_object_names=final_attachments,
                        object_position_at_close=position_at_close,
                        object_position_after_lift=avg_obj_coord.get(object_id),
                        gripper_position_at_close=gripper_at_close,
                        gripper_position_after_lift=pose.as_action()[:3],
                        **self.grasp_thresholds,
                    )
                    if grasp_outcome['outcome'] == 'grasp_verified':
                        held_object_id = object_id
                    else:
                        held_object_id = None
                    feedback = (
                        f"Grasp outcome: {grasp_outcome['outcome']} "
                        f"({grasp_outcome['reason']}). action execution: {execution_status}."
                    )
                termination_reason, primitive_postcondition_met = primitive_termination(
                    mode=mode,
                    grasp_outcome=grasp_outcome['outcome'] if grasp_outcome else None,
                    env_done=bool(done),
                    release_executed=release_executed,
                    attachment_evidence_available=attachment_evidence_available,
                    grasped_object_names=(
                        step_results[-1]['grasped_objects'] if step_results else []
                    ),
                )
                if mode == 'place' and primitive_postcondition_met:
                    held_object_id = None
                no_progress_guard.observe_execution(invocation, step_results)
                record.update({
                    'primitive': result.name,
                    'is_contact': result.is_contact,
                    'meta': result.meta,
                    'compiled_actions': [list(a) for a in result.actions],
                    'step_results': step_results,
                    'pose_after': pose.as_action(),
                    'execution_status': execution_status,
                    'status': grasp_outcome['outcome'] if grasp_outcome else execution_status,
                    'feedback': feedback,
                    'held_object_id': held_object_id,
                    'env_done': bool(done),
                    'termination_reason': termination_reason,
                    'primitive_postcondition_met': primitive_postcondition_met,
                })
                if grasp_outcome:
                    record['grasp_evidence'] = {
                        'object_position_pre_grasp': position_pre_grasp,
                        'gripper_position_pre_grasp': gripper_pre_grasp,
                        'object_position_at_close': position_at_close,
                        'object_position_after_lift': avg_obj_coord.get(object_id),
                        'gripper_position_at_close': gripper_at_close,
                        'gripper_position_after_lift': pose.as_action()[:3],
                        'attachment_at_close': attachment_at_close,
                    }
                    record['grasp_outcome'] = grasp_outcome
                history.append({'action': invocation, 'status': record['status'], 'feedback': feedback})
                trace.append(record)

            # Episode metrics.
            episode_info['instruction'] = user_instruction
            episode_info['avg_reward'] = float(np.mean(episode_info['reward'])) if episode_info['reward'] else 0.0
            episode_info['task_success'] = info['task_success']
            episode_info['num_steps'] = self.env._current_step
            episode_info['num_turns'] = turn
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            episode_info['episode_elapsed_seconds'] = info.get('episode_elapsed_seconds', 0)
            self.save_episode_metric(episode_info)
            self.save_trace(trace)
            progress_bar.update()

        self.print_task_eval_results(filename="summary.json")
        self.env.close()

    def evaluate_main(self):
        valid_eval_sets = list(self.config.get('eval_sets') or ValidEvalSets)
        if not valid_eval_sets:
            valid_eval_sets = ValidEvalSets

        for eval_set in valid_eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            real_model_name = self.model_name.split('/')[-1] if '/' in self.model_name else self.model_name
            real_model_name = real_model_name.replace(':', '_')
            exp = self.config.get('exp_name') or 'harness'
            self.log_path = 'running/eb_manipulation_harness/{}/{}/{}'.format(
                real_model_name, exp, self.eval_set
            )
            self.env = EBManEnv(
                eval_set=self.eval_set,
                render_mode=self.config.get('render_mode', 'human'),
                img_size=(self.config['resolution'], self.config['resolution']),
                down_sample_ratio=self.config['down_sample_ratio'],
                selected_indexes=list(self.config.get('selected_indexes', []) or []),
                headless=self.config.get('headless', True),
                log_path=self.log_path,
            )
            self.env._max_episode_steps = self.max_env_steps
            self.planner = HarnessPlanner(
                model_name=self.model_name,
                base_url=self.config.get('base_url'),
                api_key=self.config.get('api_key'),
                global_memory=GlobalMemory.load(self.config.get('global_memory_path', '')),
                temperature=self.config.get('temperature', 0.0),
                max_tokens=self.config.get('max_tokens', 1024),
                disable_thinking=self.config.get('disable_thinking', False),
                request_timeout=self.config.get('request_timeout', 600.0),
            )
            self.evaluate()
            with open(os.path.join(self.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))

    def check_config_valid(self):
        # Beta is language-only by construction; warn if configured otherwise.
        if not self.config.get('language_only', 1):
            logger.warning("Harness beta is language-only; forcing language_only=1.")
            self.config['language_only'] = 1
