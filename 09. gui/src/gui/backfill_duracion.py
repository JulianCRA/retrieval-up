"""Backfill duracion in the recursos table from each hash's info.json.

Usage:
    python -m gui.backfill_duracion [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from compartido.rutas import DESCARGAS_DIR, INDICE_DB


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill duracion into recursos table")
    parser.add_argument("--dry-run", action="store_true", help="Print updates without writing")
    args = parser.parse_args()

    if not INDICE_DB.exists():
        print("indice.db not found — nothing to do.")
        sys.exit(0)

    conn = sqlite3.connect(INDICE_DB)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("SELECT hash FROM recursos WHERE duracion IS NULL").fetchall()
    hashes = [r["hash"] for r in rows]
    print(f"Rows with NULL duracion: {len(hashes)}")

    updates: list[tuple[float, str]] = []
    missing = 0

    for hash_id in hashes:
        info_path = DESCARGAS_DIR / hash_id / "info.json"
        if not info_path.exists():
            missing += 1
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            missing += 1
            continue
        duracion = (info.get("descarga") or {}).get("duracion")
        if duracion is not None:
            updates.append((float(duracion), hash_id))

    print(f"  Found duration in info.json: {len(updates)}")
    print(f"  info.json missing or no duration: {missing + len(hashes) - len(updates) - missing}")
    print(f"  info.json not on disk: {missing}")

    if not updates:
        print("Nothing to update.")
        conn.close()
        return

    if args.dry_run:
        for dur, h in updates[:10]:
            print(f"  would set hash={h} duracion={dur}")
        if len(updates) > 10:
            print(f"  … and {len(updates) - 10} more")
    else:
        conn.executemany("UPDATE recursos SET duracion = ? WHERE hash = ?", updates)
        conn.commit()
        print(f"Updated {len(updates)} rows.")

    conn.close()


if __name__ == "__main__":
    main()
