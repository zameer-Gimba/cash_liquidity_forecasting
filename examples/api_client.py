"""Example client for the Agent Liquidity Forecaster API.

    python examples/api_client.py http://localhost:8000 [path/to/statement.xlsx]

Set LIQUIDITY_API_KEY if the server requires a key.
"""
from __future__ import annotations

import os
import sys

import httpx

base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
headers = {"X-API-Key": os.environ["LIQUIDITY_API_KEY"]} if os.environ.get("LIQUIDITY_API_KEY") else {}
client = httpx.Client(base_url=base, headers=headers, timeout=120)

demo = client.get("/v1/demo/forecast", params={"model": "sarima"}).raise_for_status().json()
print(f"Demo agent, {demo['date']} ({demo['weekday']}): hold ₦{demo['withdrawals']['cash_to_hold']:,.0f} cash "
      f"and ₦{demo['deposits']['float_to_hold']:,.0f} float — {demo['demand_level']} day")
for warning in demo["warnings"]:
    print("  note:", warning)

if len(sys.argv) > 2:
    with open(sys.argv[2], "rb") as fh:
        files = {"file": (os.path.basename(sys.argv[2]), fh.read())}
    report = client.post("/v1/statements/analyse", files=files).raise_for_status().json()
    print(f"\nStatement: {report['transactions']:,} transactions, {report['first_date']} to {report['last_date']}, "
          f"totals reconcile: {report['totals_match_statement']}")
    result = client.post("/v1/statements/forecast", files=files, data={"model": "sarima"}).raise_for_status().json()
    print(f"Your forecast for {result['date']}: hold ₦{result['withdrawals']['cash_to_hold']:,.0f} cash "
          f"and ₦{result['deposits']['float_to_hold']:,.0f} float")
