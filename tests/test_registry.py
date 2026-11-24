import pytest

from stride.registry import Registry


def test_builtin_registry_loads_and_converts_units():
    registry = Registry.load()
    assert "watch_fit" in registry
    app = registry.get("app_csv")
    assert app.factor("distance") == pytest.approx(1609.344)
    assert app.factor("elevation") == pytest.approx(0.3048)
    assert registry.get("watch_fit").factor("speed") == 1.0


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"name": "x", "format": "kml"}, "format"),
        ({"name": "x", "format": "gpx", "units": {"distance": "furlong"}}, "unknown distance"),
        ({"name": "x", "format": "csv"}, "map the 't'"),
        ({"name": "x", "format": "gpx", "nominal_hz": 0}, "positive"),
    ],
)
def test_invalid_profiles_are_rejected(spec, message):
    with pytest.raises(ValueError, match=message):
        Registry.from_dict({"devices": {"bad": spec}})


def test_unknown_timezone_is_rejected():
    spec = {"name": "x", "format": "gpx", "timezone": "Mars/Base"}
    with pytest.raises(ValueError, match="timezone"):
        Registry.from_dict({"devices": {"bad": spec}})


def test_unknown_device_lookup_fails():
    with pytest.raises(KeyError):
        Registry.load().get("nope")
