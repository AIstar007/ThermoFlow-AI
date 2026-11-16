import numpy as np
import pytest

from thermoprop.pennes import simulate_pennes_slab


def test_pennes_forward_simulation_returns_finite_time_depth_field():
    t, depth, field, meta = simulate_pennes_slab(duration_s=30, output_interval_s=1,
        n_depth_nodes=21, challenge_start_s=5, challenge_end_s=10)
    assert field.shape == (len(t), len(depth))
    assert field.shape[0] == 31
    assert np.isfinite(field).all()
    assert depth[0] == 0
    assert depth[-1] == pytest.approx(10.0)
    assert "forward model" in meta["model"]
    assert "not fitted" in meta["warning"]


def test_pennes_rejects_negative_perfusion():
    with pytest.raises(ValueError, match="non-negative"):
        simulate_pennes_slab(perfusion_s_inv=-0.1)
