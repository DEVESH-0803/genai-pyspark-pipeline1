"""Shared configuration and logging setup for the e-commerce pipeline."""

import logging
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"

CUSTOMERS_FILE: Path = RAW_DATA_DIR / "customers.csv"
PRODUCTS_FILE: Path = RAW_DATA_DIR / "products.csv"
ORDERS_FILE: Path = RAW_DATA_DIR / "orders.csv"

DEFAULT_CUSTOMER_COUNT: int = 500
DEFAULT_PRODUCT_COUNT: int = 100
DEFAULT_ORDER_COUNT: int = 2_000
DEFAULT_RANDOM_SEED: int = 42


def configure_logging(level: int = logging.INFO) -> None:
    """Configure consistent console logging for command-line runs."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
