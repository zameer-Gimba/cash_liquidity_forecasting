"""Smoke test: every page of the Streamlit app renders and a forecast runs."""
from __future__ import annotations
import pytest

pytest.importorskip("plotly")
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app" / "app.py")


def test_pages_render_and_forecast_runs():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    for page in ["Cash flow history", "Model accuracy", "Data", "About", "Forecast"]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
    at.selectbox[0].select(at.selectbox[0].options[0]).run()  # Random Forest: quick to train
    next(b for b in at.button if b.label == "Forecast").click().run()
    assert not at.exception and not at.error
    assert any(m.label == "Cash to hold" for m in at.metric)
