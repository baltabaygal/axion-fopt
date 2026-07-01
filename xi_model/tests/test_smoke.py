"""Smoke tests for the xi_paper reference model."""
import math

from xi_model import PredictResult, load_default_model


def test_predict_returns_sensible_result() -> None:
    model = load_default_model()
    res = model.predict(hstar=0.2, vw=0.5, theta0=1.2, beta_over_h=6.0)
    assert isinstance(res, PredictResult)
    assert res.xi > 0.0
    assert res.xi_dm > 0.0
    assert 0.0 < res.f_bm < 1.0
    assert res.g_bm >= 0.0
    assert res.rc_mean > 0.0
    assert res.tp > 0.0
    assert set(res.to_dict()) >= {"xi", "xi_dm", "f_bm", "g_bm", "rc_mean", "kappa", "tp"}


def test_xi_floor_and_high_hstar_limit() -> None:
    model = load_default_model()
    # For H_* >> M_phi the transition finishes long before oscillation: xi -> ~1.
    res = model.predict(hstar=5.0, vw=0.5, theta0=1.2, beta_over_h=6.0)
    assert res.xi >= 1.0
    assert res.xi < 2.0


def test_xi_enhances_and_decreases_with_hstar() -> None:
    model = load_default_model()
    xs = [model.predict(hstar=h, vw=0.7, theta0=1.0, beta_over_h=4.0).xi
          for h in (1e-3, 1e-2, 1e-1, 1.0)]
    # enhancement grows as the transition happens later (lower H_*)
    assert all(a > b for a, b in zip(xs, xs[1:]))
    assert xs[0] > 10.0


def test_low_hstar_bubble_misalignment_slope() -> None:
    # In the deep low-H_* (bubble-misalignment) regime xi ~ (H_*/M_phi)^{-1/2}.
    model = load_default_model()
    h1, h2 = 1e-6, 1e-5
    x1 = model.predict(hstar=h1, vw=0.7, theta0=0.1, beta_over_h=4.0).xi
    x2 = model.predict(hstar=h2, vw=0.7, theta0=0.1, beta_over_h=4.0).xi
    slope = math.log(x1 / x2) / math.log(h1 / h2)
    assert abs(slope - (-0.5)) < 0.05
