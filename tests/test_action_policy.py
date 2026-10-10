import pytest

from src.orchestration.action_policy import ActionPolicy, ActionRisk


def test_predict_digit_is_read_only():
    policy = ActionPolicy()

    assert policy.classify("predict_digit") is ActionRisk.READ_ONLY


def test_get_digit_metrics_is_read_only():
    policy = ActionPolicy()

    assert policy.classify("get_digit_metrics") is ActionRisk.READ_ONLY


def test_set_confidence_threshold_is_controlled():
    policy = ActionPolicy()

    assert (
        policy.classify("set_confidence_threshold")
        is ActionRisk.CONTROLLED
    )


def test_delete_evaluation_history_is_prohibited():
    policy = ActionPolicy()

    assert (
        policy.classify("delete_evaluation_history")
        is ActionRisk.PROHIBITED
    )


def test_unknown_action_fails_closed():
    policy = ActionPolicy()

    with pytest.raises(ValueError, match="Unknown action"):
        policy.classify("unknown_action")


def test_requires_approval_for_controlled_action():
    policy = ActionPolicy()

    assert policy.requires_approval("set_confidence_threshold") is True


def test_read_only_action_does_not_require_approval():
    policy = ActionPolicy()

    assert policy.requires_approval("predict_digit") is False


def test_is_prohibited_returns_true_for_prohibited_action():
    policy = ActionPolicy()

    assert policy.is_prohibited("delete_evaluation_history") is True