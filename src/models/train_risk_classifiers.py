"""Legacy risk model entrypoint retained for compatibility.
Liquidity_Risk is now a UI-only derived label, so this script trains no model.
"""
from __future__ import annotations

def main() -> dict[str, str]:
    return {'status': 'skipped', 'reason': 'Liquidity_Risk is UI-only and not a canonical model input'}
if __name__ == '__main__': print(main())
