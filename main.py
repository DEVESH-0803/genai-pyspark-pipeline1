"""Generate the synthetic e-commerce dataset and save it as Parquet."""

import logging
import time
from pathlib import Path

import pandas as pd

from src import config
from src.data_generator import SyntheticDataGenerator

LOGGER = logging.getLogger(__name__)


def format_size(size_bytes: int) -> str:
    """Format a file size in bytes as a readable binary unit."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def main() -> None:
    """Generate customers, products, and orders and save them as Parquet."""
    config.configure_logging()
    started_at = time.perf_counter()
    generator = SyntheticDataGenerator()
    customers, products, orders = generator.generate_all()

    outputs: tuple[tuple[str, pd.DataFrame, Path], ...] = (
        ("Customers", customers, config.RAW_DATA_DIR / "customers.parquet"),
        ("Products", products, config.RAW_DATA_DIR / "products.parquet"),
        ("Orders", orders, config.RAW_DATA_DIR / "orders.parquet"),
    )
    for label, dataframe, path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        LOGGER.info("Saved %s data to %s", label.lower(), path)

    elapsed = time.perf_counter() - started_at
    print(f"Generation and save time: {elapsed:.2f} seconds")
    for label, _, path in outputs:
        print(f"{label}: {path} ({format_size(path.stat().st_size)})")


if __name__ == "__main__":
    main()
