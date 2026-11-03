"""Categoriser and reconciliation tests on a synthetic Moniepoint statement (no real data)."""
from __future__ import annotations
import io
import pandas as pd
import pytest
from src.preprocessing import moniepoint
from src.preprocessing.moniepoint import categorise, read_statement, reconciliation, tidy_transactions
from src.preprocessing.pipeline import process_raw_statement

HOLDER = 'ACME AGENT VENTURES'

ROWS = [
    # Date, Narration, Reference, Debit, Credit, expected
    ('2022-03-01 09:00', 'WITHDRAWAL FOR card 5399', 'WTH|1', 0, 20000, 'Withdrawal'),
    ('2025-03-01 09:00', 'Purchase card 5399', 'PUR|2', 0, 15000, 'Withdrawal'),
    ('2022-03-01 10:00', 'DEPOSIT FROM *****1 TO *****2', 'DEP|3', 50000, 0, 'Deposit'),
    ('2023-05-01 10:00', 'DEPOSIT from ACME AGENT VENTURES to JOHN DOE', 'TRF|4', 30000, 0, 'Deposit'),
    ('2024-05-01 10:00', 'ACME AGENT VENTURES to JANE ROE', 'TRF|5', 40000, 0, 'Deposit'),
    ('2024-05-01 11:00', 'JOHN DOE to ACME AGENT VENTURES', 'TRF|6', 0, 25000, 'Withdrawal'),
    ('2024-05-01 11:30', 'Transfer from customer', '100004240501113000123456789012', 0, 10000, 'Withdrawal'),
    ('2024-05-01 12:00', 'Wallet Transfer from ***12 to ***34', 'TRF|7', 0, 100000, 'WalletTransfer'),
    ('2024-05-01 12:30', 'ACME AGENT VENTURES to ACME SAVINGS LTD', 'TRF|8', 60000, 0, 'WalletTransfer'),
    ('2024-05-01 13:00', 'Electronic Money Transfer Levy', 'SAV|9', 50, 0, 'Levy'),
    ('2024-05-01 13:05', 'Charge on withdrawal', 'SAV|10', 100, 0, 'Levy'),
    ('2024-05-01 13:10', 'Cashback', 'CB_CSH|11', 0, 20, 'Commission'),
    ('2024-05-01 13:20', 'Airtime MTN', 'ATP|12', 1000, 0, 'Airtime'),
    ('2024-05-01 13:30', 'Reversal of TRF|5', 'TRF|5|RVSL', 0, 40000, 'Reversal'),
    ('2024-05-01 13:40', 'DEPOSIT from JOHN DOE', 'TRF|13', 0, 5000, 'Reversal'),
]


@pytest.fixture(autouse=True)
def own_accounts(monkeypatch, tmp_path):
    monkeypatch.setattr(moniepoint, 'OWN_ACCOUNTS_FILE', tmp_path / 'none.txt')
    monkeypatch.setenv('LIQUIDITY_OWN_ACCOUNTS', 'ACME SAVINGS LTD')


def _frame():
    df = pd.DataFrame(ROWS, columns=['Date', 'Narration', 'Reference', 'Debit', 'Credit', 'Expected'])
    return tidy_transactions(df)


def test_categorise_by_code_and_side():
    df = _frame()
    got = categorise(df, HOLDER)
    wrong = df.assign(Got=got)[got != df['Expected']]
    assert wrong.empty, wrong[['Narration', 'Reference', 'Expected', 'Got']].to_string()


def test_reconciliation_matches_statement_totals():
    df = _frame()
    table = reconciliation(df, categorise(df, HOLDER))
    body = table.drop(index='TOTAL (statement)')
    assert body['Debit'].sum() == pytest.approx(table.loc['TOTAL (statement)', 'Debit'])
    assert body['Credit'].sum() == pytest.approx(table.loc['TOTAL (statement)', 'Credit'])
    assert body['Transactions'].sum() == len(df)


def test_read_statement_header_block():
    header = [['Account Statement', None, None, None, None, None],
              ['Account Name:', HOLDER, None, None, None, None],
              ['Date:', '01/03/2022 - 01/05/2024', None, None, None, None],
              [None] * 6,
              ['Date', 'Narration', 'Reference', 'Debit', 'Credit', 'Balance']]
    body = [[r[0], r[1], r[2], r[3] or None, r[4] or None, 0] for r in ROWS]
    buf = io.BytesIO()
    pd.DataFrame(header + body).to_excel(buf, header=False, index=False)
    buf.seek(0); buf.name = 'statement.xlsx'
    statement = read_statement(buf)
    assert statement.account_name == HOLDER
    assert len(statement.transactions) == len(ROWS)


def test_pipeline_daily_totals():
    df = _frame().drop(columns='Expected')
    daily = process_raw_statement(df, HOLDER)
    # Only days followed by another active day keep a next-day target.
    assert list(daily['TransactionDate']) == ['2022-03-01', '2023-05-01', '2024-05-01'][:len(daily)]
    first = daily.iloc[0]
    assert first['Withdrawal_Amount'] == 20000 and first['Deposit_Amount'] == 50000
    assert first['Target_Next_Day_Withdrawal_Amount'] == 0  # 2023-05-01 has only a deposit


def test_direction_decides_own_float():
    rows = [
        ('2024-06-01 09:00', 'ACME SAVINGS LTD to ACME AGENT VENTURES', 'TRF|20', 0, 90000, 'WalletTransfer'),
        ('2024-06-01 09:10', 'JOHN DOE to ACME SAVINGS LTD', 'TRF|21', 0, 9000, 'Withdrawal'),
        ('2024-06-01 09:20', 'ACME SAVINGS LTD to JANE ROE', 'TRF|22', 7000, 0, 'Deposit'),
        ('2024-06-01 09:30', 'Wallet funds transfer to JANE ROE', 'TRF|23', 7000, 0, 'Deposit'),
    ]
    df = tidy_transactions(pd.DataFrame(rows, columns=['Date', 'Narration', 'Reference', 'Debit', 'Credit', 'Expected']))
    assert list(categorise(df, HOLDER)) == list(df['Expected'])
