from minillm_forge.design.gpu2b_recovery import implied_capacity_mib, validate


def test_recovery_capacity_lower_bound_is_additive():
    assert implied_capacity_mib(7776, 1536) == 9312


def test_recovery_design_freeze_validates():
    result = validate()
    assert result["classification"] == "PASS", result["errors"]
    assert result["training_executed"] is False
