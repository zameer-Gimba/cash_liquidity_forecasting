"""REST API tests (synthetic data only)."""
from __future__ import annotations
import datetime as dt
import io

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _daily_totals(days: int = 120) -> list[dict]:
    rng = np.random.default_rng(0)
    start = dt.date(2025, 1, 1)
    return [{"date": (start + dt.timedelta(days=i)).isoformat(),
             "withdrawals": float(500_000 + 200_000 * ((i % 7) == 5) + rng.normal(0, 50_000)),
             "deposits": float(450_000 + rng.normal(0, 40_000))} for i in range(days)]


def _statement_xlsx(days: int = 90) -> bytes:
    """A small Moniepoint-style export: header block, then the transaction table."""
    rows, start = [], dt.datetime(2025, 3, 1, 9)
    for i in range(days):
        day = start + dt.timedelta(days=i)
        rows += [
            [day, "Withdrawal card 5399", f"PUR|{i}a", None, 40_000 + 1_000 * (i % 7), None],
            [day + dt.timedelta(hours=1), "JOHN DOE to ACME AGENT VENTURES", f"TRF|{i}b", None, 20_000, None],
            [day + dt.timedelta(hours=2), "ACME AGENT VENTURES to JANE ROE", f"TRF|{i}c", 55_000, None, None],
            [day + dt.timedelta(hours=3), "Electronic Money Transfer Levy", f"SAV|{i}d", 50, None, None],
        ]
    header = [["Account Statement", None, None, None, None, None],
              ["Account Name:", "ACME AGENT VENTURES", None, None, None, None],
              [None] * 6,
              ["Date", "Narration", "Reference", "Debit", "Credit", "Balance"]]
    buf = io.BytesIO()
    pd.DataFrame(header + rows).to_excel(buf, header=False, index=False)
    return buf.getvalue()


def test_health_and_models():
    assert client.get("/health").json()["status"] == "ok"
    body = client.get("/v1/models").json()
    assert body["default"] == "sarima"
    assert {m["key"] for m in body["models"]} == {"sarima", "arima", "rf", "xgb", "lgbm"}


def test_demo_forecast():
    r = client.get("/v1/demo/forecast", params={"date": "2026-01-20", "include_path": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["horizon_days"] == 4 and body["warnings"] == []
    w = body["withdrawals"]
    assert w["cash_to_hold"] == pytest.approx(w["expected"] * 1.15, rel=1e-6)
    assert body["typical_error"]["mae"] > 0
    assert body["path"][-1]["date"] == "2026-01-20"


def test_demo_forecast_far_future_warns():
    body = client.get("/v1/demo/forecast", params={"date": "2026-11-05", "model": "rf"}).json()
    assert body["horizon_days"] > 30 and body["warnings"]


def test_forecast_from_daily_totals():
    rows = _daily_totals()
    r = client.post("/v1/forecast", json={"daily": rows, "date": "2025-05-05"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["data"]["trading_days"] == 120
    assert 0 < body["withdrawals"]["expected"] < 2_000_000
    assert body["deposits"]["method"].startswith("SARIMA")


def test_daily_totals_validation():
    assert client.post("/v1/forecast", json={"daily": _daily_totals(30)}).status_code == 422
    assert client.post("/v1/forecast", json={"daily": _daily_totals(), "model": "rf"}).status_code == 422
    r = client.post("/v1/forecast", json={"daily": _daily_totals(), "date": "2025-01-10"})
    assert r.status_code == 422 and "after the last data date" in r.json()["detail"]


def test_statement_analyse_reconciles():
    files = {"file": ("statement.xlsx", _statement_xlsx(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    r = client.post("/v1/statements/analyse", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["totals_match_statement"] is True
    cats = {c["category"]: c for c in body["categories"]}
    assert cats["Withdrawal"]["transactions"] == 180  # card + transfer cash-outs
    assert cats["Deposit"]["money_out"] == pytest.approx(90 * 55_000)
    assert "ACME" not in r.text  # no names in the response


def test_statement_forecast():
    files = {"file": ("statement.xlsx", _statement_xlsx(), "application/octet-stream")}
    r = client.post("/v1/statements/forecast", files=files, data={"date": "2025-06-05", "model": "rf"})
    assert r.status_code == 200, r.text
    assert r.json()["data"]["source"] == "uploaded statement"


def test_bad_file_rejected():
    r = client.post("/v1/statements/analyse", files={"file": ("x.csv", b"foo,bar\n1,2\n", "text/csv")})
    assert r.status_code == 422


def test_api_key(monkeypatch):
    monkeypatch.setenv("LIQUIDITY_API_KEYS", "secret-1,secret-2")
    rows = {"daily": _daily_totals()}
    assert client.post("/v1/forecast", json=rows).status_code == 401
    assert client.post("/v1/forecast", json=rows, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/v1/forecast", json=rows, headers={"X-API-Key": "secret-2"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_public_demo(monkeypatch):
    from api import main
    monkeypatch.setenv("LIQUIDITY_API_KEYS", "secret-1")
    main._hits.clear()
    assert client.get("/v1/models").status_code == 200
    assert client.get("/v1/demo/forecast", params={"date": "2026-01-20"}).status_code == 200
    assert client.get("/v1/demo/forecast", params={"date": "2026-01-20", "model": "rf"}).status_code == 401
    assert client.get("/v1/demo/forecast", params={"date": "2026-01-20", "model": "lgbm"},
                      headers={"X-API-Key": "secret-1"}).status_code == 200


def test_public_rate_limit(monkeypatch):
    from api import main
    monkeypatch.setenv("LIQUIDITY_API_KEYS", "secret-1")
    main._hits.clear()
    codes = [client.get("/v1/models", headers={"X-Forwarded-For": "203.0.113.9"}).status_code for _ in range(main.PUBLIC_RATE_LIMIT + 1)]
    assert codes[:-1] == [200] * main.PUBLIC_RATE_LIMIT and codes[-1] == 429
    assert client.get("/v1/models", headers={"X-Forwarded-For": "203.0.113.9", "X-API-Key": "secret-1"}).status_code == 200
    main._hits.clear()
