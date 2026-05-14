from xi_model import load_default_model


def test_smoke_prediction() -> None:
    model = load_default_model()
    result = model.predict(hstar=0.2, vw=0.5, theta0=1.308997, beta_over_h=10.0)
    assert result.xi > 0.0
    assert result.xi_hi >= result.xi
    assert result.xi_lo <= result.xi
    assert result.xi_dm_mode == "broken_powerlaw_ftilde"
    assert result.prediction_status == "validated"
    assert result.status_flags == []
    assert model.xi_dm_grid_source == "xi_dm_broken_powerlaw"


def test_low_hstar_plateau_policy() -> None:
    model = load_default_model()
    result = model.predict(hstar=0.003, vw=0.5, theta0=1.2, beta_over_h=6.0, clip=True)
    ref = model.predict(hstar=0.05, vw=0.5, theta0=1.2, beta_over_h=6.0, clip=True)
    assert result.kappa_plateau_applied is True
    assert result.kappa_plateau_hstar > result.clipped_inputs["hstar"]
    assert result.geometry_vw == 0.5
    assert result.geometry_hstar == min(model.geom_h_grid)
    assert result.tp > ref.tp
    assert result.prediction_status == "continued"
    assert "continued_low_h_kappa_plateau" in result.status_flags
    assert "continued_low_h_geometry_boundary" in result.status_flags
    assert any("kappa_pilot frozen to low-H plateau" in w for w in result.warnings)


def test_high_hstar_geometry_boundary_policy() -> None:
    model = load_default_model()
    result = model.predict(hstar=3.0, vw=0.5, theta0=1.2, beta_over_h=6.0, clip=False)
    assert result.geometry_hstar == max(model.geom_h_grid)
    assert result.prediction_status == "continued"
    assert "continued_high_h_geometry_boundary" in result.status_flags
    assert any("geometry evaluated at high-H boundary" in w for w in result.warnings)


def test_vw_extrapolation_to_one_uses_geometry_support() -> None:
    model = load_default_model()
    result = model.predict(hstar=0.05, vw=1.0, theta0=1.2, beta_over_h=6.0, clip=True)
    assert result.geometry_vw == 1.0
    assert result.clipped_inputs["vw"] == 1.0
    assert result.xi > 0.0
    assert result.prediction_status in {"continued", "validated_with_generated_geometry"}
    assert "continued_outside_fit_domain" in result.status_flags
