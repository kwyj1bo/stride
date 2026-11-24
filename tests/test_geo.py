import numpy as np
import pytest

from stride.geo import (
    WGS84_A,
    LocalFrame,
    ecef_to_geodetic,
    geodetic_to_ecef,
    haversine_m,
    path_length,
)


def test_ecef_round_trip_is_sub_millimetre():
    lat = np.array([0.0, 19.076, -33.9, 64.1, 89.0])
    lon = np.array([0.0, 72.8777, 151.2, -21.9, 10.0])
    h = np.array([0.0, 14.0, 58.0, 2000.0, -30.0])
    back = ecef_to_geodetic(*geodetic_to_ecef(lat, lon, h))
    np.testing.assert_allclose(back[0], lat, atol=1e-9)
    np.testing.assert_allclose(back[1], lon, atol=1e-9)
    np.testing.assert_allclose(back[2], h, atol=1e-3)


def test_enu_axes_point_east_north_up():
    frame = LocalFrame(0.0, 0.0, 0.0)
    east = frame.to_enu([0.0], [0.001])[0]
    north = frame.to_enu([0.001], [0.0])[0]
    up = frame.to_enu([0.0], [0.0], [100.0])[0]
    # one milli-degree of longitude on the equator is a * dlambda
    assert east[0] == pytest.approx(WGS84_A * np.radians(0.001), rel=1e-6)
    assert abs(east[1]) < 1e-6
    assert north[1] == pytest.approx(110.574, abs=0.01)  # meridian arc at the equator
    np.testing.assert_allclose(up, [0, 0, 100], atol=1e-6)


def test_enu_round_trip_and_distance_matches_haversine():
    frame = LocalFrame(19.076, 72.8777, 10.0)
    lat = 19.076 + np.linspace(0, 0.01, 50)
    lon = 72.8777 + np.linspace(0, 0.01, 50)
    enu = frame.to_enu(lat, lon, np.full(50, 10.0))
    back_lat, back_lon, _ = frame.to_geodetic(enu)
    np.testing.assert_allclose(back_lat, lat, atol=1e-9)
    np.testing.assert_allclose(back_lon, lon, atol=1e-9)
    sphere = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum()
    assert path_length(enu) == pytest.approx(sphere, rel=5e-3)


def test_frame_anchors_at_first_valid_fix_and_skips_nan():
    lat = np.array([np.nan, 19.0, 19.001])
    lon = np.array([np.nan, 72.0, 72.0])
    frame = LocalFrame.at_first_fix(lat, lon)
    enu = frame.to_enu(lat, lon)
    assert np.isnan(enu[0]).all()
    np.testing.assert_allclose(enu[1], 0, atol=1e-6)
    assert path_length(enu) == pytest.approx(110.7, abs=0.2)
