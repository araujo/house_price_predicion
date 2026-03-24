"""Smoke test against running API (stub CLI)."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    _args = parser.parse_args()
    # Implementation: httpx GET /health
    print("smoke_test_api: not implemented")


if __name__ == "__main__":
    main()
