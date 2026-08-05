"""Simulator-free checks for manipulation evaluator phase integration."""

import json
import sys
import types
from types import SimpleNamespace

import pytest

from embodiedbench.planner.harness.global_memory import GlobalMemory


@pytest.fixture
def evaluator_class(monkeypatch):
    stubs = {
        'embodiedbench.envs.eb_manipulation.EBManEnv': {
            'EBManEnv': object,
            'ValidEvalSets': ['base'],
        },
        'embodiedbench.envs.eb_manipulation.eb_man_utils': {
            'form_harness_grounding_artifact_for_input': lambda *args, **kwargs: None,
        },
        'embodiedbench.envs.eb_manipulation.rgbd_grounding': {
            'compute_oracle_metrics': lambda *args: {},
            'summarize_oracle_frames': lambda frames: {},
        },
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop(
        'embodiedbench.evaluator.eb_manipulation_harness_evaluator', None
    )
    from embodiedbench.evaluator.eb_manipulation_harness_evaluator import (
        EB_ManipulationHarnessEvaluator,
    )

    return EB_ManipulationHarnessEvaluator


def _config(**overrides):
    config = {
        'model_name': 'fake',
        'protocol_phase': 'deployment',
        'bootstrap_seed': 7,
        'evaluation_seeds': [101, 202],
        'bootstrap_budget': 12,
        'deployment_budget': 4,
        'selected_indexes': [0, 12],
        'episode_protocol_seeds': [101, 202],
        'max_env_steps': 30,
    }
    config.update(overrides)
    return config


def test_legacy_protocol_is_unspecified_and_keeps_budget(evaluator_class):
    evaluator = evaluator_class({
        'model_name': 'fake', 'max_env_steps': 30
    })

    assert evaluator.protocol_phase == 'unspecified'
    assert evaluator.phase_manifest is None
    assert evaluator.max_env_steps == 30
    assert evaluator._protocol_audit()['reportable'] is True


def test_opt_in_seed_mapping_is_rejected_before_environment_creation(evaluator_class):
    evaluator = evaluator_class.__new__(evaluator_class)
    evaluator.config = _config(episode_protocol_seeds=[101])
    evaluator.max_env_steps = 30
    evaluator.configured_max_env_steps = 30
    evaluator.phase_manifest = None
    evaluator.phase_policy = None
    evaluator.episode_protocol_seeds = ()

    with pytest.raises(ValueError, match='align 1:1'):
        evaluator._configure_phase_protocol()


def test_phase_budget_caps_environment_steps_and_never_expands_config(evaluator_class):
    deployment = evaluator_class(_config())
    bootstrap = evaluator_class(_config(
        protocol_phase='bootstrap',
        selected_indexes=[9],
        episode_protocol_seeds=[7],
        max_env_steps=3,
    ))

    assert deployment.max_env_steps == 4
    assert bootstrap.max_env_steps == 3


def test_deployment_memory_writer_is_blocked_before_fake_writer(evaluator_class):
    evaluator = evaluator_class(_config())
    calls = []

    with pytest.raises(ValueError, match='memory writes are forbidden'):
        evaluator.write_memory(101, lambda: calls.append('called'))

    assert calls == []


def test_bootstrap_episode_is_kept_but_excluded_from_summary(
    tmp_path, evaluator_class
):
    results = tmp_path / 'results'
    results.mkdir()
    (results / 'episode_1_res.json').write_text(json.dumps({
        'task_success': 1,
        'planner_steps': 2,
        'protocol': {'phase': 'bootstrap', 'reportable': False},
    }), encoding='utf-8')
    evaluator = evaluator_class(_config(
        protocol_phase='bootstrap',
        selected_indexes=[9],
        episode_protocol_seeds=[7],
    ))
    evaluator.env = SimpleNamespace(log_path=str(tmp_path))
    evaluator.log_path = str(tmp_path)

    evaluator.print_task_eval_results('summary.json')

    summary = json.loads((results / 'summary.json').read_text(encoding='utf-8'))
    assert (results / 'episode_1_res.json').exists()
    assert summary['total_num_tasks'] == 0
    assert summary['num_success'] == 0


def test_deployment_hashes_use_loaded_content_and_must_not_change(evaluator_class):
    evaluator = evaluator_class(_config())
    memory = GlobalMemory.seeded()
    evaluator.planner = SimpleNamespace(global_memory=memory)
    evaluator._protocol_memory_before = evaluator._memory_hashes(memory)

    evaluator._verify_deployment_memory_unchanged()
    assert evaluator._protocol_memory_after == evaluator._protocol_memory_before

    memory.success_rules.append('mutated')
    with pytest.raises(RuntimeError, match='deployment memory changed'):
        evaluator._verify_deployment_memory_unchanged()