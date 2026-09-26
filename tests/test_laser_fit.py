import json

import numpy as np
import pytest

from act.laser import LaserFit, fit_poly


def truth(xy: np.ndarray) -> np.ndarray:
    x, y = xy[..., 0], xy[..., 1]
    pan = 1200 + 6.1 * x - 0.8 * y + 0.021 * x * x - 0.013 * x * y + 0.004 * y * y
    tilt = 1400 + 0.5 * x + 5.3 * y - 0.006 * x * x + 0.009 * x * y + 0.031 * y * y
    return np.stack([pan, tilt], axis=-1)


def grid_pts(n=5) -> np.ndarray:
    return np.array([(x, y) for x in np.linspace(0, 90, n) for y in np.linspace(0, 60, n)])


def test_fit_recovers_known_mapping():
    xy = grid_pts()
    fit = fit_poly(xy, truth(xy), grid=5)
    assert np.abs(fit.predict(xy) - truth(xy)).max() < 0.5
    off = np.random.default_rng(1).uniform((0, 0), (90, 60), (50, 2))
    assert np.abs(fit.predict(off) - truth(off)).max() < 0.5
    assert fit.predict((45.0, 30.0)).shape == (2,)
    assert fit.fit_error_cm["max"] < 0.01 and fit.n_points == 25


def test_jacobian_matches_finite_differences():
    xy = grid_pts()
    fit = fit_poly(xy, truth(xy))
    for p in [(10.0, 5.0), (45.0, 30.0), (80.0, 55.0)]:
        J = fit.jacobian(p)
        h = 1e-3
        fd = np.stack([(fit.predict((p[0] + h, p[1])) - fit.predict((p[0] - h, p[1]))) / (2 * h),
                       (fit.predict((p[0], p[1] + h)) - fit.predict((p[0], p[1] - h))) / (2 * h)], axis=1)
        assert np.allclose(J, fd, atol=1e-5)


def test_fit_error_reported_in_cm():
    xy = grid_pts()
    p = truth(xy) + np.random.default_rng(0).normal(0, 3, (len(xy), 2))   # ~3 µs noise
    fit = fit_poly(xy, p)
    assert 0.05 < fit.fit_error_cm["median"] < 1.0                         # ~6 µs/cm here


def test_save_load_roundtrip(tmp_path):
    xy = grid_pts()
    fit = fit_poly(xy, truth(xy), grid=4)
    path = tmp_path / "sub" / "laser_cal.json"
    fit.save(str(path))
    d = json.loads(path.read_text())
    assert {"coef", "fit_error_cm", "timestamp", "grid", "n_points"} <= set(d)
    back = LaserFit.load(str(path))
    assert back.grid == 4 and back.n_points == 25
    assert np.allclose(back.predict(xy), fit.predict(xy))
    assert back.fit_error_cm == fit.fit_error_cm


def test_too_few_points():
    xy = grid_pts(2)
    with pytest.raises(ValueError):
        fit_poly(xy, truth(xy))
