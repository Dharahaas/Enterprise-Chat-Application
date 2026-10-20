"""Provision an employee or administrator account without storing passwords in source."""

from __future__ import annotations

import argparse
from getpass import getpass

from app.auth import create_user, initialize_auth_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an Enterprise Knowledge Assistant user.")
    parser.add_argument("username", help="Employee username or email address")
    parser.add_argument("--role", choices=("employee", "admin"), default="employee")
    args = parser.parse_args()

    password = getpass("New password (minimum 12 characters): ")
    confirmation = getpass("Confirm password: ")
    if password != confirmation:
        parser.error("The passwords do not match.")

    initialize_auth_store()
    try:
        create_user(args.username, password, args.role)
    except ValueError as error:
        parser.error(str(error))
    print(f"Created {args.role} account for {args.username.strip().casefold()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
