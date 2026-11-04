"""REST API for the Agent Liquidity Forecaster.

Run locally:
    uvicorn api.main:app --reload

Interactive documentation: http://localhost:8000/docs

Authentication: set LIQUIDITY_API_KEYS to a comma-separated list of keys and
send one in the ``X-API-Key`` header. When the variable is unset the API is
open (for local use and demos).

Statements are processed in memory and never stored or logged.
"""
from __future__ import annotations

import datetime as dt
import hmac
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.forecast_service import (
    DEFAULT_MODEL,
    MODELS,
    STATESPACE,
    ForecastError,
    dataset_from_canonical,
    dataset_from_totals,
    demo_dataset,
    forecast,
    test_results,
    warm_up,
)
from src.preprocessing.moniepoint import categorise, reconciliation, tidy_transactions
from src.preprocessing.pipeline import load_input, process_raw_statement, validate_schema
from src.utils.config import DATE_COL
from src.utils.dataset_detector import attempt_auto_repair, detect_dataset_type

API_VERSION = "1.0.0"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ModelKey = Literal["sarima", "arima", "rf", "xgb", "lgbm"]

CATEGORY_LABELS = {
    "Withdrawal": "Customer cash-outs (card + transfer)",
    "Deposit": "Customer cash deposits",
    "WalletTransfer": "Own float movements",
    "Levy": "Fees, levies and charges",
    "Commission": "Cashback and rewards",
    "Reversal": "Reversals and refunds",
    "Airtime": "Airtime and data",
    "BillPayment": "Bill payments",
    "Insurance": "Insurance",
    "Other": "Loans and terminal events",
}

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Pre-train the demo models in the background; the server accepts requests meanwhile.
    threading.Thread(target=warm_up, daemon=True).start()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Agent Liquidity Forecaster API",
    version=API_VERSION,
    description=(
        "Forecast how much **physical cash** and **float** a POS agent needs on a future date, "
        "from the agent's own Moniepoint statement or daily totals.\n\n"
        "* `cash_to_hold` = expected customer cash-outs + 15% buffer\n"
        "* `float_to_hold` = expected customer deposits + 10% buffer\n\n"
        "Amounts are in naira (NGN). Statements are processed in memory and never stored."
    ),
    contact={"name": "Muhammad Ibrahim Gimba", "url": "https://github.com/zameer-Gimba/cash_liquidity_forecasting"},
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: str | None = Security(_api_key_header)) -> None:
    keys = [k.strip() for k in os.environ.get("LIQUIDITY_API_KEYS", "").split(",") if k.strip()]
    if keys and not any(key and hmac.compare_digest(key, k) for k in keys):
        raise HTTPException(status_code=401, detail="Missing or invalid API key (X-API-Key header).")


@app.exception_handler(ForecastError)
def _forecast_error(_: Request, exc: ForecastError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.middleware("http")
async def _timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.0f}"
    return response


# ------------------------------------------------------------------ schemas


class DailyTotal(BaseModel):
    date: dt.date
    withdrawals: float = Field(ge=0, description="Customer cash-outs that day (card + transfer cash-out), NGN")
    deposits: float = Field(ge=0, description="Customer cash deposits that day, NGN")


class ForecastRequest(BaseModel):
    daily: list[DailyTotal] = Field(min_length=60, description="One row per trading day, at least 60 days")
    date: dt.date | None = Field(None, description="Forecast date; defaults to tomorrow")
    model: Literal["sarima", "arima"] = DEFAULT_MODEL
    include_path: bool = Field(False, description="Also return daily forecasts for up to 14 days ending on the date")

    model_config = {"json_schema_extra": {"example": {
        "daily": [{"date": "2025-11-01", "withdrawals": 850000, "deposits": 910000}, "... at least 60 days ..."],
        "date": "2025-12-31", "model": "sarima", "include_path": False,
    }}}


class Withdrawals(BaseModel):
    expected: float
    safety_buffer: float
    cash_to_hold: float


class Deposits(BaseModel):
    expected: float
    safety_buffer: float
    float_to_hold: float
    method: str


class DataInfo(BaseModel):
    source: str
    trading_days: int
    first_date: str
    last_date: str


class PathPoint(BaseModel):
    date: str
    expected_withdrawals: float


class TypicalError(BaseModel):
    mae: float = Field(description="Mean absolute error on the held-out test period, NGN")
    vs_best_simple_rule_pct: float


class ForecastResponse(BaseModel):
    date: str
    weekday: str
    model: str
    data: DataInfo
    horizon_days: int
    withdrawals: Withdrawals
    deposits: Deposits
    demand_level: Literal["quiet", "normal", "busy"]
    demand_level_basis: str
    warnings: list[str]
    currency: str
    typical_error: TypicalError | None = None
    path: list[PathPoint] | None = None


class CategoryTotal(BaseModel):
    category: str
    label: str
    transactions: int
    money_in: float
    money_out: float


class MonthTotal(BaseModel):
    month: str
    withdrawals: float
    deposits: float


class AnalysisResponse(BaseModel):
    transactions: int
    first_date: str
    last_date: str
    trading_days: int
    totals_match_statement: bool
    categories: list[CategoryTotal]
    statement_totals: dict[str, float]
    monthly: list[MonthTotal]


# ------------------------------------------------------------------ helpers


def _read_upload(file: UploadFile) -> io.BytesIO:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 50 MB.")
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    buffer = io.BytesIO(data)
    buffer.name = file.filename or "statement.xlsx"
    return buffer


def _split_names(own_accounts: str | None) -> list[str]:
    return [n.strip() for n in (own_accounts or "").split(",") if n.strip()]


def _statement(file: UploadFile, own_accounts: str | None):
    """(daily dataset, transactions, categories) for an uploaded statement or engineered CSV."""
    try:
        raw = load_input(_read_upload(file))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - unreadable spreadsheet
        raise HTTPException(status_code=400, detail=f"Could not read the file: {exc}") from exc
    kind = detect_dataset_type(raw)
    names = _split_names(own_accounts)
    if kind == "RAW_STATEMENT":
        holder = raw.attrs.get("account_name", "")
        tx = tidy_transactions(raw)
        if tx.empty:
            raise ForecastError("The statement has no transactions.")
        return process_raw_statement(raw, holder, names), tx, categorise(tx, holder, names)
    if kind == "FEATURE_ENGINEERED":
        return validate_schema(raw), None, None
    if kind == "PARTIALLY_ENGINEERED":
        return validate_schema(attempt_auto_repair(raw)[0]), None, None
    raise ForecastError("Unrecognised file. Upload a Moniepoint statement export (.xlsx) or an engineered liquidity CSV.")


# ------------------------------------------------------------------ endpoints


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health", tags=["Service"], summary="Health check")
def health() -> dict:
    return {"status": "ok", "version": API_VERSION}


@app.get("/v1/models", tags=["Service"], summary="Available models and their tested accuracy")
def models(_: None = Depends(require_key)) -> dict:
    results = test_results()
    return {
        "default": DEFAULT_MODEL,
        "models": [
            {"key": key, "name": name, "needs_full_statement": key not in STATESPACE, "test": results.get(name)}
            for key, name in MODELS.items()
        ],
        "simple_rules": {k.replace("Baseline: ", ""): v for k, v in results.items() if k.startswith("Baseline")},
        "test_period": "2025-06-07 to 2026-01-16 (demo agent, never used for training)",
    }


@app.get("/v1/demo/forecast", response_model=ForecastResponse, response_model_exclude_none=True, tags=["Forecast"],
         summary="Forecast for the demo agent")
def demo_forecast(
    date: dt.date | None = Query(None, description="Forecast date; defaults to tomorrow"),
    model: ModelKey = Query(DEFAULT_MODEL),
    include_path: bool = Query(False),
    _: None = Depends(require_key),
) -> dict:
    """Uses the anonymised demo agent (daily totals from a real Moniepoint agent, Dec 2021 – Jan 2026)."""
    return forecast(demo_dataset(), date, model, include_path)


@app.post("/v1/forecast", response_model=ForecastResponse, response_model_exclude_none=True, tags=["Forecast"],
          summary="Forecast from daily totals")
def forecast_from_totals(body: ForecastRequest, _: None = Depends(require_key)) -> dict:
    """Send daily withdrawal and deposit totals (at least 60 days). Only the time-series models are
    available here, because the other features need the full statement."""
    rows = [r.model_dump() for r in body.daily]
    return forecast(dataset_from_totals(rows), body.date, body.model, body.include_path)


@app.post("/v1/statements/forecast", response_model=ForecastResponse, response_model_exclude_none=True,
          tags=["Forecast"], summary="Forecast from a Moniepoint statement")
def forecast_from_statement(
    file: UploadFile = File(..., description="Moniepoint statement export (.xlsx) or engineered liquidity CSV"),
    date: dt.date | None = Form(None, description="Forecast date; defaults to tomorrow"),
    model: ModelKey = Form(DEFAULT_MODEL),
    own_accounts: str | None = Form(None, description="Your other account names, comma-separated"),
    include_path: bool = Form(False),
    _: None = Depends(require_key),
) -> dict:
    daily, _tx, _cat = _statement(file, own_accounts)
    return forecast(dataset_from_canonical(daily, "uploaded statement"), date, model, include_path)


@app.post("/v1/statements/analyse", response_model=AnalysisResponse, tags=["Statements"],
          summary="Categorise a statement and check it reconciles")
def analyse_statement(
    file: UploadFile = File(..., description="Moniepoint statement export (.xlsx)"),
    own_accounts: str | None = Form(None, description="Your other account names, comma-separated"),
    _: None = Depends(require_key),
) -> dict:
    """Every transaction is put in exactly one category; the category totals must equal the statement
    totals. Returns counts and amounts only, no names or account numbers."""
    daily, tx, categories = _statement(file, own_accounts)
    if tx is None:
        raise ForecastError("Analysis needs a raw Moniepoint statement, not an engineered CSV.")
    table = reconciliation(tx, categories)
    totals = table.loc["TOTAL (statement)"]
    body = table.drop(index="TOTAL (statement)")
    rows = [
        {"category": cat, "label": CATEGORY_LABELS.get(cat, cat), "transactions": int(r.Transactions),
         "money_in": round(float(r.Credit), 2), "money_out": round(float(r.Debit), 2)}
        for cat, r in body.iterrows()
    ]
    monthly = (
        daily.assign(month=pd.to_datetime(daily[DATE_COL]).dt.strftime("%Y-%m"))
        .groupby("month")[["Withdrawal_Amount", "Deposit_Amount"]].sum()
    )
    return {
        "transactions": int(len(tx)),
        "first_date": f"{tx['Date'].min():%Y-%m-%d}",
        "last_date": f"{tx['Date'].max():%Y-%m-%d}",
        "trading_days": int(tx["Date"].dt.normalize().nunique()),
        "totals_match_statement": bool(abs(body["Debit"].sum() - totals["Debit"]) < 1
                                       and abs(body["Credit"].sum() - totals["Credit"]) < 1),
        "categories": rows,
        "statement_totals": {"money_in": round(float(totals["Credit"]), 2), "money_out": round(float(totals["Debit"]), 2)},
        "monthly": [{"month": m, "withdrawals": round(float(r.Withdrawal_Amount), 2), "deposits": round(float(r.Deposit_Amount), 2)}
                    for m, r in monthly.iterrows()],
    }
