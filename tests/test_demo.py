from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "demo" / "app.py")


def price_shown(at: AppTest) -> float:
    return float(at.metric[0].value.replace("$", "").replace(",", ""))


def test_demo_renders_a_price():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert 200_000 < price_shown(at) < 5_000_000


def test_more_bedrooms_and_space_costs_more():
    at = AppTest.from_file(APP, default_timeout=60).run()
    base = price_shown(at)
    inputs = {ni.label: ni for ni in at.number_input}
    inputs["Bedrooms"].set_value(5)
    inputs["Building area (m²)"].set_value(320)
    at.run()
    assert not at.exception
    assert price_shown(at) > base


def test_changing_suburb_updates_location():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox[0].select("Toorak").run()
    assert not at.exception
    assert at.text_input[0].value == "Stonnington City Council"
