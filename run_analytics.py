"""Load generated Parquet data and run the sales analytics interactively."""

import logging
import time

from src import config
from src.spark_analytics import SalesAnalytics

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run and time each Spark load and analysis, then stop Spark cleanly."""
    config.configure_logging()

    started = time.perf_counter()
    spark = SalesAnalytics.create_spark_session()
    print(f"Create Spark session: {time.perf_counter() - started:.2f} seconds")

    try:
        load_operations = (
            ("customers", config.RAW_DATA_DIR / "customers.parquet"),
            ("products", config.RAW_DATA_DIR / "products.parquet"),
            ("orders", config.RAW_DATA_DIR / "orders.parquet"),
        )
        loaded = {}
        for name, path in load_operations:
            operation_started = time.perf_counter()
            loaded[name] = SalesAnalytics.load_parquet(path, spark)
            print(f"Load {name}: {time.perf_counter() - operation_started:.2f} seconds")

        operation_started = time.perf_counter()
        top_customers = SalesAnalytics.top_customers_by_revenue(
            loaded["orders"], loaded["products"]
        )
        top_customers.show(truncate=False)
        print(f"Top customers by revenue: {time.perf_counter() - operation_started:.2f} seconds")

        operation_started = time.perf_counter()
        top_customers_by_quantity = SalesAnalytics.top_customers_by_quantity(
            loaded["orders"], n=20
        )
        top_customers_by_quantity.show(truncate=False)
        print(
            "Top customers by quantity: "
            f"{time.perf_counter() - operation_started:.2f} seconds"
        )

        operation_started = time.perf_counter()
        category_sales = SalesAnalytics.sales_by_category(loaded["orders"], loaded["products"])
        category_sales.show(truncate=False)
        print(f"Sales by category: {time.perf_counter() - operation_started:.2f} seconds")

        operation_started = time.perf_counter()
        monthly_trends = SalesAnalytics.monthly_trends(loaded["orders"], loaded["products"])
        monthly_trends.show(truncate=False)
        print(f"Monthly trends: {time.perf_counter() - operation_started:.2f} seconds")
    except Exception:
        LOGGER.exception("Analytics run failed")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
