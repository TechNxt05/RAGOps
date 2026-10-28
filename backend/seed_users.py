"""
Create the admin account from ADMIN_EMAIL / ADMIN_PASSWORD (same as app startup).

    ADMIN_EMAIL=you@company.com ADMIN_PASSWORD='a-strong-password' python seed_users.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.auth.bootstrap import bootstrap_admin  # noqa: E402
from app.db import init_db  # noqa: E402

if __name__ == "__main__":
    init_db()
    bootstrap_admin()
    print("Done. (Nothing is created unless ADMIN_EMAIL and ADMIN_PASSWORD are set.)")
