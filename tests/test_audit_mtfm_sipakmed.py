from server_audit_mtfm_sipakmed import validate_config_contract


def test_auditor_accepts_the_locked_200_epoch_contract():
    source = {"run_config": {"epochs": 200, "batch_size": 32, "lr": 0.02, "momentum": 0.9, "weight_decay": 5e-4, "warmup_epochs": 3, "lambda_manual": 0.5, "alpha": 0.1, "beta": 0.6, "seed": 42}}
    assert validate_config_contract(source) == []


def test_auditor_rejects_an_unapproved_epoch_change():
    source = {"run_config": {"epochs": 100, "batch_size": 32, "lr": 0.02, "momentum": 0.9, "weight_decay": 5e-4, "warmup_epochs": 3, "lambda_manual": 0.5, "alpha": 0.1, "beta": 0.6, "seed": 42}}
    assert any("epochs" in failure for failure in validate_config_contract(source))
