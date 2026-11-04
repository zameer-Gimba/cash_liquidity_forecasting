"""Read and categorise Moniepoint agent account statements.

Moniepoint has changed how it records the same business event several
times (e.g. customer cash deposits were ``DEP`` until early 2023, then
``TRF`` transfers with changing narrations; card cash-outs moved from
``WTH`` to ``PUR`` in 2025). Matching on narration text alone therefore
silently drops whole categories. This module classifies primarily by the
transaction-type code at the start of the ``Reference`` (``WTH|...``,
``TRF|...``) plus the debit/credit side, and uses the narration only to
separate the agent's own float movements from customer transactions.

Categories (what they mean for an agent's liquidity):

- Withdrawal      customer cash-out: card withdrawals and transfer cash-outs
                  (customer sends money to the agent, collects physical cash)
- Deposit         customer cash deposit: agent receives cash and transfers
                  the money out from the wallet
- WalletTransfer  the agent's own float movements between own wallets and
                  accounts (top-ups, sweeps) - not customer demand
- Levy            per-transaction fees, charges, levies and stamp duty
- Commission      cashback and rewards credited to the agent
- Airtime, BillPayment, Insurance, Reversal
- Other           loans and terminal events (activation, swap, contributions)

Own accounts: transfers to or from the statement holder's other accounts
are float movements. The holder's name is read from the statement header;
other own-account names are read at runtime from
``config/own_accounts.txt`` (one name per line, git-ignored) or the
``LIQUIDITY_OWN_ACCOUNTS`` environment variable (comma-separated), so no
personal names live in the code.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import ROOT_DIR

OWN_ACCOUNTS_FILE = ROOT_DIR / "config" / "own_accounts.txt"

CARD_CASHOUT_CODES = {"WTH", "PUR"}
INBOUND_TRANSFER_CODES = {"TRF", "AP_TRSF", "APR_TRSF", "MIT"}
FLOAT_CODES = {"FND", "WFT"}
FEE_CODES = {"SAV", "USSD_CHARGE"}
COMMISSION_CODES = {"CB_CSH", "CB_CSH_OUT", "ARF", "RCT"}
AIRTIME_CODES = {"ATP", "DTP"}
BILL_CODES = {"BPT"}
INSURANCE_CODES = {"ISP"}
OTHER_CODES = {"LTT", "LPT", "WCL", "TAC", "TST", "MPT", "OP_DEP"}

# Transfers between the agent's own wallets / linked accounts, where both sides are
# masked account numbers (no customer name), e.g. "Wallet Transfer from ***12 to ***34".
FLOAT_NARRATION = re.compile(r"^(WALLET TRANSFER|TRANSFER|CARD_TRANSFER) FROM [\d*]+ TO [\d*]+|^WALLET TRANSFER$|^SELF_CREDIT")
LEVY_NARRATION = re.compile(r"LEVY|VALUE ADDED TAX|STAMP DUTY|USSD CHARGE")

STATEMENT_COLUMNS = ["Date", "Narration", "Reference", "Debit", "Credit", "Balance"]


@dataclass
class Statement:
    """A parsed statement: transactions plus header details."""

    transactions: pd.DataFrame
    account_name: str = ""
    period: str = ""
    warnings: list[str] = field(default_factory=list)


def _normalise(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def own_account_keywords(holder_name: str = "", own_accounts: list[str] | None = None) -> list[str]:
    """Normalised names whose transfers count as the agent's own float."""

    names = []
    if holder_name:
        names.append(holder_name)
    if OWN_ACCOUNTS_FILE.exists():
        names += [line.strip() for line in OWN_ACCOUNTS_FILE.read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.startswith("#")]
    names += [n.strip() for n in os.environ.get("LIQUIDITY_OWN_ACCOUNTS", "").split(",") if n.strip()]
    names += [n.strip() for n in (own_accounts or []) if n and n.strip()]
    keys = {_normalise(n) for n in names}
    # Also match the first two words alone (bank apps often truncate names).
    for name in names:
        words = re.findall(r"[A-Za-z0-9]+", name)
        if len(words) >= 2:
            keys.add(_normalise(" ".join(words[:2])))
    return sorted(k for k in keys if len(k) >= 6)


def _read_raw(source) -> pd.DataFrame:
    """Read an xlsx/xls/csv statement as strings, tolerant of odd styling."""

    name = getattr(source, "name", str(source)).lower()
    if name.endswith(".csv"):
        return pd.read_csv(source, header=None, dtype=str)
    data = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    for engine in ("calamine", "openpyxl"):
        try:
            return pd.read_excel(io.BytesIO(data), header=None, dtype=str, engine=engine)
        except Exception:  # noqa: BLE001 - try the next engine
            continue
    raise ValueError("Could not read the statement file (tried calamine and openpyxl).")


def read_statement(source) -> Statement:
    """Parse a Moniepoint statement export (header block + transaction table)."""

    raw = _read_raw(source)
    header_row = None
    for i in range(min(50, len(raw))):
        cells = {str(v).strip().lower() for v in raw.iloc[i].tolist() if isinstance(v, str)}
        if {"date", "narration", "reference", "debit", "credit"} <= cells:
            header_row = i
            break
    if header_row is None:
        raise ValueError("No transaction header (Date, Narration, Reference, Debit, Credit) found.")

    meta = {}
    for i in range(header_row):
        values = [str(v).strip() for v in raw.iloc[i].tolist() if isinstance(v, str) and str(v).strip()]
        if len(values) >= 2 and values[0].endswith(":"):
            meta[values[0].rstrip(":").lower()] = values[1]

    headers = [str(v).strip() if isinstance(v, str) else "" for v in raw.iloc[header_row].tolist()]
    table = raw.iloc[header_row + 1:].copy()
    table.columns = headers
    table = table.loc[:, [c for c in STATEMENT_COLUMNS if c in table.columns]]
    return Statement(
        transactions=tidy_transactions(table),
        account_name=meta.get("account name", ""),
        period=meta.get("date", ""),
    )


def tidy_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Typed, de-duplicated transactions with a ``Code`` column."""

    df = df.copy()
    for column in ("Debit", "Credit", "Balance"):
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["Debit"] = df["Debit"].fillna(0.0)
    df["Credit"] = df["Credit"].fillna(0.0)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[(df["Debit"] != 0) | (df["Credit"] != 0)]
    df["Narration"] = df["Narration"].fillna("").astype(str)
    df["Reference"] = df["Reference"].fillna("").astype(str)
    df["Code"] = df["Reference"].str.split("|").str[0].str.strip().str.upper()
    # Some inbound bank transfers carry a bare numeric session id as reference.
    df.loc[df["Code"].str.match(r"^\d{12,}"), "Code"] = "BANK_TRANSFER_IN"
    df = df.drop_duplicates(subset=["Date", "Reference", "Debit", "Credit"], keep="first")
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def categorise(df: pd.DataFrame, holder_name: str = "", own_accounts: list[str] | None = None) -> pd.Series:
    """Liquidity category for every transaction (see module docstring)."""

    keys = own_account_keywords(holder_name, own_accounts)
    narration = df["Narration"].str.upper().str.strip()
    code = df["Code"]
    is_credit = df["Credit"] > 0
    is_debit = ~is_credit

    # "SENDER TO RECIPIENT": money arriving is own float only if an own account sent it;
    # money leaving is own float only if an own account receives it. Customers paying the
    # agent name the agent as recipient, so a bare mention of an own name is not enough.
    has_to = narration.str.contains(" TO ", regex=False)
    sender = narration.str.split(" TO ", n=1).str[0]
    recipient = narration.str.split(" TO ", n=1).str[1].fillna("")
    own_float = (
        (has_to & is_credit & _mentions(sender, keys))
        | (has_to & is_debit & _mentions(recipient, keys))
        | (~has_to & _mentions(narration, keys))
    )
    floaty = narration.str.match(FLOAT_NARRATION)
    levy = narration.str.contains(LEVY_NARRATION)
    reversal = df["Reference"].str.upper().str.contains("RVSL")

    category = pd.Series("Other", index=df.index, dtype=object)
    # MPC-ITR-TRF-...: inter-bank transfer postings (older export format).
    transfer_like = code.isin(INBOUND_TRANSFER_CODES | {"BANK_TRANSFER_IN"}) | code.str.startswith("MPC-ITR-TRF")

    rules = [
        (reversal, "Reversal"),
        (levy | code.isin(FEE_CODES), "Levy"),
        (code.isin(COMMISSION_CODES), "Commission"),
        (code.isin(AIRTIME_CODES), "Airtime"),
        (code.isin(BILL_CODES), "BillPayment"),
        (code.isin(INSURANCE_CODES), "Insurance"),
        (code.isin(OTHER_CODES), "Other"),
        (code.isin(FLOAT_CODES), "WalletTransfer"),
        (code.isin(CARD_CASHOUT_CODES) & is_credit, "Withdrawal"),
        ((code == "DEP") & is_debit, "Deposit"),
        # Transfers: the agent's own float first ...
        (transfer_like & (floaty | own_float), "WalletTransfer"),
        # ... a refunded / failed deposit coming back
        (transfer_like & is_credit & narration.str.startswith("DEPOSIT FROM"), "Reversal"),
        # ... customers sending money to collect cash
        (transfer_like & is_credit, "Withdrawal"),
        # ... the agent sending a customer's cash deposit onward
        (transfer_like & is_debit, "Deposit"),
    ]

    assigned = pd.Series(False, index=df.index)
    for mask, label in rules:
        take = mask & ~assigned
        category[take] = label
        assigned |= take
    return category


def _mentions(text: pd.Series, keys: list[str]) -> pd.Series:
    """True where the text names one of the own-account keywords (spacing/punctuation ignored)."""
    if not keys:
        return pd.Series(False, index=text.index)
    return text.map(_normalise).str.contains("|".join(map(re.escape, keys)), regex=True)


def reconciliation(df: pd.DataFrame, categories: pd.Series) -> pd.DataFrame:
    """Totals per category; the debit and credit columns sum to the statement totals."""

    table = (
        df.assign(Category=categories)
        .groupby("Category")
        .agg(Transactions=("Category", "size"), Debit=("Debit", "sum"), Credit=("Credit", "sum"))
        .sort_values("Transactions", ascending=False)
    )
    totals = pd.DataFrame(
        {"Transactions": [len(df)], "Debit": [df["Debit"].sum()], "Credit": [df["Credit"].sum()]},
        index=["TOTAL (statement)"],
    )
    return pd.concat([table, totals])
