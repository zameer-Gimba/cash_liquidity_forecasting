"""
Streamlit dashboard for the Agent Liquidity Forecaster.

Run locally:
    streamlit run streamlit_app/app.py

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.models.predict_deposit import predict_deposit  # noqa: E402
from src.models.session_forecast import (  # noqa: E402
    MODEL_OPTIONS,
    build_future_feature_row,
    forecast_path,
    predict_withdrawal_for_date,
    train_withdrawal_model,
)
from src.preprocessing.moniepoint import categorise, reconciliation  # noqa: E402
from src.preprocessing.pipeline import load_input, process_raw_statement, validate_schema  # noqa: E402
from src.utils.config import (  # noqa: E402
    CANONICAL_DATASET_PATH,
    CANONICAL_TRAINING_END_DATE,
    DATE_COL,
    MODEL_DIR,
    ROOT_DIR,
    SAFETY_BUFFER_DEPOSIT,
    SAFETY_BUFFER_WITHDRAWAL,
)
from src.utils.dataset_detector import attempt_auto_repair, detect_dataset_type  # noqa: E402

APP_TITLE = "Agent Liquidity Forecaster"
GITHUB_URL = "https://github.com/zameer-Gimba/cash_liquidity_forecasting"

WITHDRAWAL_COLOR = "#2A78D6"   # cash paid out to customers
DEPOSIT_COLOR = "#E08A2E"      # float used for customer deposits
FORECAST_COLOR = "#D03B3B"
REFERENCE_GRAY = "#898781"

RESULTS_DIR = ROOT_DIR / "reports" / "results"
RECOMMENDED_MODEL = "SARIMA"
MODEL_NOTES = {
    "SARIMA": "most accurate in testing",
    "ARIMA": "",
    "Random Forest": "",
    "XGBoost": "",
    "LightGBM": "",
    "LSTM": "slow; least accurate in testing",
}
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
    "Transfer": "Unclassified transfers",
}

st.set_page_config(page_title=APP_TITLE, page_icon=str(PROJECT_ROOT / "assets" / "favicon.png"), layout="wide")


# ------------------------------------------------------------------ data & state


@st.cache_data(show_spinner=False)
def canonical_dataset() -> pd.DataFrame:
    df = pd.read_csv(CANONICAL_DATASET_PATH)
    # After mid-January 2026 the account was barely used; those days are not normal trading.
    return df[df[DATE_COL] <= CANONICAL_TRAINING_END_DATE].reset_index(drop=True)


def init_state() -> None:
    defaults = {"dataset": None, "source": "Demo agent", "dataset_id": "canonical",
                "forecast": None, "upload_report": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state.dataset is None:
        st.session_state.dataset = canonical_dataset()


def use_dataset(df: pd.DataFrame, source: str, dataset_id: str) -> None:
    st.session_state.update(dataset=df, source=source, dataset_id=dataset_id, forecast=None)


@st.cache_resource(show_spinner=False, max_entries=12)
def fitted_model(model_key: str, dataset_id: str, _df: pd.DataFrame):
    return train_withdrawal_model(model_key, _df)


@st.cache_resource(show_spinner=False)
def deposit_models():
    try:
        return joblib.load(MODEL_DIR / "rf_deposit_classifier.pkl"), joblib.load(MODEL_DIR / "rf_deposit_regressor.pkl")
    except Exception:  # noqa: BLE001 - deposit forecast is optional
        return None


def history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    return out.sort_values(DATE_COL).reset_index(drop=True)


def naira(value: float) -> str:
    return f"₦{value:,.0f}"


def short_naira(value: float) -> str:
    for size, suffix in ((1e9, "bn"), (1e6, "m")):
        if abs(value) >= size:
            return f"₦{value / size:,.2f}{suffix}"
    return naira(value)


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(namelength=-1),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
        yaxis=dict(tickprefix="₦", separatethousands=True),
    )
    return fig


def results(name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def model_error(label: str) -> float | None:
    table = results("withdrawal_model_comparison.csv")
    if table is None or "MAE" not in table:
        return None
    row = table[table["Model"] == label]
    return float(row["MAE"].iloc[0]) if len(row) and pd.notna(row["MAE"].iloc[0]) else None


# ------------------------------------------------------------------ pages


def page_forecast() -> None:
    st.title(APP_TITLE)
    st.caption(
        "How much physical cash and how much float will a POS agent need on a given day? "
        "Pick a date and the forecaster suggests what to hold, with a safety buffer, "
        "based on the agent's own transaction history."
    )

    data = history(st.session_state.dataset)
    last_date = data[DATE_COL].max()
    tomorrow = dt.date.today() + dt.timedelta(days=1)
    first_allowed = (last_date + pd.Timedelta(days=1)).date()

    labels = [f"{name} ({MODEL_NOTES[name]})" if MODEL_NOTES.get(name) else name for name in MODEL_OPTIONS]
    c1, c2, c3 = st.columns([1, 1.4, 0.8])
    target = c1.date_input(
        "Forecast date",
        value=max(tomorrow, first_allowed),
        min_value=first_allowed,
        max_value=max(tomorrow, first_allowed) + dt.timedelta(days=365),
    )
    choice = c2.selectbox("Model", labels, index=list(MODEL_OPTIONS).index(RECOMMENDED_MODEL))
    model_label = list(MODEL_OPTIONS)[labels.index(choice)]
    c3.write("")
    c3.write("")
    run = c3.button("Forecast", type="primary", width="stretch")

    horizon = (pd.Timestamp(target) - last_date).days
    note = (
        f"Data: **{st.session_state.source}**, {len(data):,} trading days up to "
        f"**{last_date:%d %b %Y}**. This forecast is **{horizon} day{'s' if horizon != 1 else ''}** past the last data."
    )
    if horizon > 30:
        st.warning(
            note + " Far ahead, the forecast only carries the weekly pattern and the last known level, "
            "so it cannot see changes since then. Upload a recent statement on the **Data** page "
            "for an up-to-date forecast."
        )
    else:
        st.caption(note)

    if run:
        model_key = MODEL_OPTIONS[model_label]
        with st.spinner(f"Training {model_label} on {len(data):,} days..."):
            try:
                model, extra = fitted_model(model_key, st.session_state.dataset_id, st.session_state.dataset)
                withdrawal = predict_withdrawal_for_date(model_key, model, extra, data, pd.Timestamp(target))
                deposit = None
                models = deposit_models()
                if models is not None:
                    deposit = predict_deposit(build_future_feature_row(data, pd.Timestamp(target)), *models)
                path = None
                if model_key != "lstm":
                    if horizon <= 60:
                        dates = pd.date_range(last_date + pd.Timedelta(days=1), pd.Timestamp(target))
                    else:
                        dates = pd.date_range(pd.Timestamp(target) - pd.Timedelta(days=13), pd.Timestamp(target))
                    path = forecast_path(model_key, model, extra, data, list(dates))
                st.session_state.forecast = dict(
                    date=target, model=model_label, withdrawal=max(withdrawal, 0.0), deposit=deposit, path=path,
                    dataset_id=st.session_state.dataset_id,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Forecast failed: {exc}")

    fc = st.session_state.forecast
    if not fc or fc["dataset_id"] != st.session_state.dataset_id:
        st.info(f"Choose a date and click **Forecast**. {RECOMMENDED_MODEL} is selected because it was the most accurate model in testing.")
        recent_chart(data, None)
        return

    st.subheader(f"{pd.Timestamp(fc['date']):%A %d %B %Y} · {fc['model']}")
    recent = data.tail(90)["Withdrawal_Amount"]
    q25, q75 = recent.quantile([0.25, 0.75])
    level = "Busy" if fc["withdrawal"] > q75 else "Quiet" if fc["withdrawal"] < q25 else "Normal"
    icon = {"Busy": "⚠️", "Quiet": "💤", "Normal": "✅"}[level]

    cols = st.columns(4)
    cols[0].metric("Cash to hold", naira(fc["withdrawal"] * (1 + SAFETY_BUFFER_WITHDRAWAL)),
                   help=f"Expected cash-outs plus a {SAFETY_BUFFER_WITHDRAWAL:.0%} safety buffer.")
    cols[1].metric("Expected cash-outs", naira(fc["withdrawal"]))
    if fc["deposit"] == 0:
        cols[2].metric("Float to hold", "₦0", help="The deposit model expects no customer deposits on this day.")
    elif fc["deposit"] is not None:
        cols[2].metric("Float to hold", naira(fc["deposit"] * (1 + SAFETY_BUFFER_DEPOSIT)),
                       help=f"Expected customer deposits plus a {SAFETY_BUFFER_DEPOSIT:.0%} safety buffer.")
    else:
        cols[2].metric("Float to hold", "n/a")
    cols[3].metric("Demand", f"{icon} {level}", help="Compared with the last 90 trading days in the data.")

    error = model_error(fc["model"])
    if error and st.session_state.dataset_id == "canonical":
        st.caption(
            f"In testing on Jun 2025 – Jan 2026, {fc['model']}'s forecasts were off by {naira(error)} on a typical day. "
            "Day-to-day cash demand is noisy, which is why the buffer matters."
        )
    recent_chart(data, fc)


def recent_chart(data: pd.DataFrame, fc: dict | None) -> None:
    recent = data.tail(90)
    far = bool(fc) and (pd.Timestamp(fc["date"]) - recent[DATE_COL].max()).days > 60
    fig = go.Figure()
    fig.add_bar(x=recent[DATE_COL], y=recent["Withdrawal_Amount"], name="Cash-outs (actual)",
                marker_color=WITHDRAWAL_COLOR, opacity=0.55,
                hovertemplate="%{x|%a %d %b %Y}<br>₦%{y:,.0f}<extra>Actual</extra>")
    fig.add_scatter(x=recent[DATE_COL], y=recent["Withdrawal_Amount"].rolling(7, min_periods=1).mean(),
                    name="7-day average", line=dict(color=REFERENCE_GRAY, width=2),
                    hovertemplate="%{x|%d %b %Y}<br>₦%{y:,.0f}<extra>7-day average</extra>")
    if fc and not far:
        add_forecast(fig, fc)
    fig.update_layout(title="Daily customer cash-outs: last 90 trading days" + (" and forecast" if fc and not far else ""))
    if not far:
        st.plotly_chart(style(fig, 420), width="stretch")
        return
    left, right = st.columns([1.6, 1])
    left.plotly_chart(style(fig, 400), width="stretch")
    week = go.Figure()
    add_forecast(week, fc)
    week.update_layout(title="Forecast: the two weeks up to the date", showlegend=False)
    right.plotly_chart(style(week, 400), width="stretch")


def add_forecast(fig: go.Figure, fc: dict) -> None:
    if fc.get("path") is not None and len(fc["path"]):
        fig.add_scatter(x=fc["path"].index, y=fc["path"].values, name="Forecast", mode="lines+markers",
                        line=dict(color=FORECAST_COLOR, width=2, dash="dot"), marker=dict(size=6),
                        hovertemplate="%{x|%a %d %b %Y}<br>₦%{y:,.0f}<extra>Forecast</extra>")
    fig.add_scatter(x=[pd.Timestamp(fc["date"])], y=[fc["withdrawal"]], name="Forecast date", mode="markers",
                    marker=dict(size=15, color=FORECAST_COLOR, symbol="star", line=dict(color="white", width=1)),
                    hovertemplate="%{x|%a %d %b %Y}<br>₦%{y:,.0f}<extra>Forecast date</extra>")


def page_history() -> None:
    st.title("Cash flow history")
    st.caption("What the agent's money actually did, day by day. Cash-outs drain physical cash; deposits drain float.")
    data = history(st.session_state.dataset)

    start, end = data[DATE_COL].min().date(), data[DATE_COL].max().date()
    picked = st.slider("Period", min_value=start, max_value=end, value=(start, end), format="MMM YYYY")
    view = data[(data[DATE_COL].dt.date >= picked[0]) & (data[DATE_COL].dt.date <= picked[1])]
    if view.empty:
        st.info("No trading days in this period.")
        return

    cols = st.columns(4)
    cols[0].metric("Trading days", f"{len(view):,}")
    cols[1].metric("Customer cash-outs", short_naira(view["Withdrawal_Amount"].sum()))
    cols[2].metric("Customer deposits", short_naira(view["Deposit_Amount"].sum()))
    cols[3].metric("Average cash-outs per day", naira(view["Withdrawal_Amount"].mean()))

    monthly = view.set_index(DATE_COL)[["Withdrawal_Amount", "Deposit_Amount"]].resample("MS").sum()
    fig = go.Figure()
    fig.add_bar(x=monthly.index, y=monthly["Withdrawal_Amount"], name="Cash-outs", marker_color=WITHDRAWAL_COLOR,
                hovertemplate="%{x|%b %Y}<br>₦%{y:,.0f}<extra>Cash-outs</extra>")
    fig.add_bar(x=monthly.index, y=monthly["Deposit_Amount"], name="Deposits", marker_color=DEPOSIT_COLOR,
                hovertemplate="%{x|%b %Y}<br>₦%{y:,.0f}<extra>Deposits</extra>")
    fig.update_layout(title="Monthly totals", barmode="group", bargap=0.25)
    st.plotly_chart(style(fig), width="stretch")

    left, right = st.columns(2)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday = view.groupby("DayOfWeek")[["Withdrawal_Amount", "Deposit_Amount"]].mean().reindex(range(7))
    fig = go.Figure()
    fig.add_bar(x=days, y=weekday["Withdrawal_Amount"], name="Cash-outs", marker_color=WITHDRAWAL_COLOR,
                hovertemplate="%{x}<br>₦%{y:,.0f}<extra>Average cash-outs</extra>")
    fig.add_bar(x=days, y=weekday["Deposit_Amount"], name="Deposits", marker_color=DEPOSIT_COLOR,
                hovertemplate="%{x}<br>₦%{y:,.0f}<extra>Average deposits</extra>")
    fig.update_layout(title="Average day by weekday", barmode="group")
    left.plotly_chart(style(fig, 340), width="stretch")

    totals = pd.DataFrame(
        [(CATEGORY_LABELS.get(c, c), float(view[f"{c}_Amount"].sum()), int(view[f"{c}_Count"].sum()))
         for c in CATEGORY_LABELS if f"{c}_Amount" in view],
        columns=["Category", "Amount", "Transactions"],
    )
    totals = totals[totals["Transactions"] > 0].sort_values("Amount")
    fig = go.Figure(go.Bar(x=totals["Amount"], y=totals["Category"], orientation="h", marker_color=WITHDRAWAL_COLOR,
                           customdata=totals["Transactions"],
                           hovertemplate="%{y}<br>₦%{x:,.0f} · %{customdata:,} transactions<extra></extra>"))
    fig.update_layout(title="Money by category", xaxis=dict(tickprefix="₦", separatethousands=True))
    fig = style(fig, 340)
    fig.update_layout(yaxis=dict(tickprefix=""), xaxis_type="log")
    right.plotly_chart(fig, width="stretch")
    right.caption("Log scale, so small categories stay visible.")


def page_accuracy() -> None:
    st.title("Model accuracy")
    st.caption(
        "Every model was tested on the last 15% of trading days (7 Jun 2025 – 16 Jan 2026), which it never saw "
        "during training. Each one is compared with simple rules a person could use without any model."
    )
    table = results("withdrawal_model_comparison.csv")
    if table is None:
        st.warning("Run `python -m src.models.evaluate_models` to generate the comparison.")
        return

    table = table.dropna(subset=["MAE"]).copy()
    baselines = table[table["Model"].str.startswith("Baseline")]
    best = baselines.loc[baselines["MAE"].idxmin()]
    table["vs best rule"] = (table["MAE"] / best["MAE"] - 1) * 100
    table["Kind"] = table["Model"].str.startswith("Baseline").map({True: "Simple rule", False: "Model"})
    table["Name"] = table["Model"].str.replace("Baseline: ", "", regex=False)
    ordered = table.sort_values("MAE")
    winner = ordered[ordered["Kind"] == "Model"].iloc[0]

    cols = st.columns(3)
    cols[0].metric("Most accurate model", winner["Name"])
    cols[1].metric("Typical daily error", naira(winner["MAE"]))
    cols[2].metric("Versus best simple rule", f"{winner['vs best rule']:+.1f}%", delta_color="inverse",
                   help=f"Best simple rule: {best['Model'].replace('Baseline: ', '')} ({naira(best['MAE'])}).")

    fig = go.Figure(go.Bar(
        x=ordered["MAE"], y=ordered["Name"], orientation="h",
        marker_color=[REFERENCE_GRAY if k == "Simple rule" else WITHDRAWAL_COLOR for k in ordered["Kind"]],
        customdata=ordered[["Kind", "vs best rule"]],
        hovertemplate="%{y} (%{customdata[0]})<br>Typical error ₦%{x:,.0f}<br>%{customdata[1]:+.1f}% vs best rule<extra></extra>",
    ))
    fig.add_vline(x=best["MAE"], line=dict(color=REFERENCE_GRAY, dash="dash"))
    fig.update_layout(title="Typical daily error for cash-outs (shorter is better; grey = simple rules)",
                      yaxis=dict(autorange="reversed"), xaxis=dict(tickprefix="₦", separatethousands=True))
    fig = style(fig, 380)
    fig.update_layout(yaxis=dict(tickprefix="", autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"""
**What this means.** Day-to-day cash demand at one agent is mostly noise around a weekly rhythm and a
slowly changing level. {winner['Name']}, which models exactly that, beats the best simple rule by
{abs(winner['vs best rule']):.0f}%. The machine-learning models roughly match the 7-day average; the LSTM
needs far more data than one agent has. More agents and outside signals (market days, salary dates,
holidays) are the most likely way to do better.
"""
    )

    with st.expander("Full tables"):
        show = table[["Name", "Kind", "MAE", "RMSE", "R2", "vs best rule"]].sort_values("MAE")
        st.markdown("**Cash-outs (withdrawals)**")
        st.dataframe(show.style.format({"MAE": "₦{:,.0f}", "RMSE": "₦{:,.0f}", "R2": "{:.3f}", "vs best rule": "{:+.1f}%"}),
                     hide_index=True, width="stretch")
        deposits = results("deposit_regressor_comparison.csv")
        if deposits is not None:
            st.markdown("**Deposit amount**")
            st.dataframe(deposits.drop(columns=["Error"], errors="ignore").dropna(subset=["MAE"])
                         .style.format({"MAE": "₦{:,.0f}", "RMSE": "₦{:,.0f}", "MAPE": "{:.1f}%", "R2": "{:.3f}"}),
                         hide_index=True, width="stretch")
        classifier = results("deposit_classifier_comparison.csv")
        if classifier is not None:
            st.markdown("**Will there be a deposit tomorrow?** (this agent has deposits almost every day, so this is near-trivial)")
            st.dataframe(classifier.drop(columns=["ROC_AUC"], errors="ignore"), hide_index=True, width="stretch")


def page_data() -> None:
    st.title("Data")
    st.caption(
        "The demo uses one real Moniepoint agent's history (daily totals only, no names). "
        "Upload your own Moniepoint statement to forecast for your business. Files are processed "
        "in memory and never stored."
    )
    cols = st.columns(3)
    data = history(st.session_state.dataset)
    cols[0].metric("Active data", st.session_state.source)
    cols[1].metric("Trading days", f"{len(data):,}")
    cols[2].metric("Period", f"{data[DATE_COL].min():%b %Y} – {data[DATE_COL].max():%b %Y}")

    uploaded = st.file_uploader("Moniepoint statement (.xlsx) or an engineered liquidity CSV", type=["xlsx", "xls", "csv"])
    own = st.text_input("Your other account names (optional, comma-separated)",
                        help="Transfers to or from these accounts count as moving your own float, not customer demand.")
    b1, b2 = st.columns(2)
    if b1.button("Process file", type="primary", disabled=uploaded is None, width="stretch"):
        try:
            with st.spinner("Reading and categorising transactions..."):
                process_upload(uploaded, own)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not process this file: {exc}")
    if b2.button("Back to demo data", disabled=st.session_state.dataset_id == "canonical", width="stretch"):
        use_dataset(canonical_dataset(), "Demo agent", "canonical")
        st.session_state.upload_report = None
        st.rerun()

    report = st.session_state.upload_report
    if report is not None and st.session_state.dataset_id != "canonical":
        st.subheader("Check: every naira accounted for")
        table = report.copy()
        table.index = [CATEGORY_LABELS.get(i, i) for i in table.index]
        body = table.drop(index="TOTAL (statement)")
        ok = abs(body["Debit"].sum() - table.loc["TOTAL (statement)", "Debit"]) < 1 and \
            abs(body["Credit"].sum() - table.loc["TOTAL (statement)", "Credit"]) < 1
        (st.success if ok else st.error)(
            "Category totals match the statement totals." if ok else "Category totals do not match the statement totals."
        )
        st.dataframe(table.rename(columns={"Debit": "Money out", "Credit": "Money in"})
                     .style.format({"Money out": "₦{:,.0f}", "Money in": "₦{:,.0f}", "Transactions": "{:,}"}),
                     width="stretch")

    st.download_button("Download processed daily dataset (CSV)", st.session_state.dataset.to_csv(index=False).encode(),
                       "liquidity_dataset.csv", "text/csv")


def process_upload(uploaded, own_accounts: str) -> None:
    raw = load_input(uploaded)
    kind = detect_dataset_type(raw)
    report = None
    if kind == "RAW_STATEMENT":
        holder = raw.attrs.get("account_name", "")
        names = [n.strip() for n in own_accounts.split(",") if n.strip()]
        clean = process_raw_statement(raw, holder, names)
        from src.preprocessing.moniepoint import tidy_transactions
        tx = tidy_transactions(raw)
        report = reconciliation(tx, categorise(tx, holder, names))
    elif kind == "FEATURE_ENGINEERED":
        clean = validate_schema(raw)
    elif kind == "PARTIALLY_ENGINEERED":
        clean = validate_schema(attempt_auto_repair(raw)[0])
    else:
        raise ValueError("Unrecognised format. Upload a Moniepoint statement export or an engineered liquidity CSV.")
    if len(clean) < 60:
        raise ValueError(f"Only {len(clean)} trading days found; at least 60 are needed to forecast.")
    dataset_id = f"upload:{uploaded.name}:{len(clean)}:{clean[DATE_COL].iloc[-1]}:{own_accounts}"
    use_dataset(clean, f"Uploaded: {uploaded.name}", dataset_id)
    st.session_state.upload_report = report
    st.success(f"Ready: {len(clean):,} trading days. Go to **Forecast** to use it.")


def page_about() -> None:
    st.title("About")
    st.markdown(f"""
POS agents run out of money in two ways: not enough **physical cash** when customers cash out, or not
enough **float** (wallet balance) when customers deposit or transfer. Both mean turned-away customers.
This tool forecasts both from the agent's own statement and suggests how much of each to hold.

**How it works**

1. **Read the statement.** Moniepoint `.xlsx` exports are parsed past their header block.
2. **Categorise every transaction** by its type code (the start of the reference, such as `WTH`, `PUR`,
   `DEP`, `TRF`) and whether money came in or went out. Customer transfer cash-outs and card cash-outs
   are withdrawals; transfers out for customers' cash are deposits; moves between the agent's own
   accounts are kept apart. The category totals must equal the statement totals.
3. **Daily features:** totals per category, weekday, recent levels (lags and rolling averages).
4. **Forecast** with SARIMA (weekly seasonality), ARIMA, Random Forest, XGBoost, LightGBM or an LSTM.
5. **Recommend** cash to hold = expected cash-outs + {SAFETY_BUFFER_WITHDRAWAL:.0%}; float to hold =
   expected deposits + {SAFETY_BUFFER_DEPOSIT:.0%}.

**Limitations.** The demo data is one agent in one location, so results may not carry over to other agents
without their own data. Day-to-day demand is noisy: the best model beats a simple 7-day average by
about 7%, not by a wide margin. Forecasts far past the last statement date can't see what changed since.
The model doesn't yet know about holidays, market days or salary dates.

**API:** the same forecasts are available as a REST API for apps and partners: [API documentation](https://cash-liquidity-api.onrender.com/docs) (key required).

**Author:** Muhammad Ibrahim Gimba · [GitHub]({GITHUB_URL})
""")


# ------------------------------------------------------------------ main


def main() -> None:
    init_state()
    pages = {
        "Forecast": page_forecast,
        "Cash flow history": page_history,
        "Model accuracy": page_accuracy,
        "Data": page_data,
        "About": page_about,
    }
    with st.sidebar:
        st.header("Navigation")
        choice = st.radio("Go to", list(pages), label_visibility="collapsed")
        st.divider()
        data = st.session_state.dataset
        st.caption(f"**{st.session_state.source}**\n\n{len(data):,} trading days · "
                   f"{pd.to_datetime(data[DATE_COL]).min():%b %Y} – {pd.to_datetime(data[DATE_COL]).max():%b %Y}")
    pages[choice]()


main()
