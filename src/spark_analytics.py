"""Sales analytics for the synthetic e-commerce Parquet datasets."""

import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src import config

LOGGER = logging.getLogger(__name__)


class SalesAnalytics:
    """Load e-commerce data and calculate revenue and sales trends with Spark."""

    @staticmethod
    def create_spark_session(app_name: str = "EcommerceSalesAnalytics") -> SparkSession:
        """Create a local Spark session with 4 GB memory, AQE, and Kryo enabled."""
        return (
            SparkSession.builder.appName(app_name)
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )

    @staticmethod
    def load_parquet(path: str | Path, spark: SparkSession | None = None) -> DataFrame:
        """Load a Parquet file or directory into a Spark DataFrame.

        If no session is passed, the active/default Spark session is used. Create
        one first with ``create_spark_session`` when explicit local settings are needed.
        """
        parquet_path = Path(path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet path does not exist: {parquet_path}")
        session = spark or SparkSession.getActiveSession() or SparkSession.getDefaultSession()
        if session is None:
            raise RuntimeError("No active Spark session. Call create_spark_session() first.")
        LOGGER.info("Loading Parquet data from %s", parquet_path)
        return session.read.parquet(str(parquet_path))

    @staticmethod
    def top_customers_by_revenue(
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 5,
    ) -> DataFrame:
        """Return the top N customers by completed-order revenue (customer spend)."""
        if n < 1:
            raise ValueError("n must be a positive integer")
        joined = orders_df.join(products_df, on="product_id", how="inner")
        return (
            joined.withColumn("revenue", F.col("quantity") * F.col("price"))
            .groupBy("customer_id")
            .agg(F.round(F.sum("revenue"), 2).alias("total_spend"))
            .orderBy(F.desc("total_spend"))
            .limit(n)
        )

    @staticmethod
    def top_customers_by_quantity(orders_df: DataFrame, n: int = 10) -> DataFrame:
        """Return the top N customers ranked by the total units they ordered."""
        if n < 1:
            raise ValueError("n must be a positive integer")
        return (
            orders_df.groupBy("customer_id")
            .agg(F.sum("quantity").alias("total_quantity"))
            .orderBy(F.desc("total_quantity"))
            .limit(n)
        )

    @staticmethod
    def sales_by_category(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Summarize revenue and units sold for each product category."""
        joined = orders_df.join(products_df, on="product_id", how="inner")
        return (
            joined.withColumn("revenue", F.col("quantity") * F.col("price"))
            .groupBy("category")
            .agg(
                F.round(F.sum("revenue"), 2).alias("total_revenue"),
                F.sum("quantity").alias("units_sold"),
            )
            .orderBy(F.desc("total_revenue"))
        )

    @staticmethod
    def monthly_trends(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Return monthly revenue and month-over-month growth percentage.

        The first month has a null growth value because no prior month exists.
        """
        joined = orders_df.join(products_df, on="product_id", how="inner")
        monthly_revenue = (
            joined.withColumn("month", F.date_trunc("month", F.to_timestamp("order_date")))
            .withColumn("revenue", F.col("quantity") * F.col("price"))
            .groupBy("month")
            .agg(F.round(F.sum("revenue"), 2).alias("total_revenue"))
        )
        month_window = Window.orderBy("month")
        with_previous_month = monthly_revenue.withColumn(
            "previous_month_revenue", F.lag("total_revenue").over(month_window)
        )
        return (
            with_previous_month.withColumn(
                "revenue_growth_pct",
                F.when(
                    F.col("previous_month_revenue").isNotNull()
                    & (F.col("previous_month_revenue") != 0),
                    F.round(
                        (F.col("total_revenue") - F.col("previous_month_revenue"))
                        / F.col("previous_month_revenue")
                        * 100,
                        2,
                    ),
                ),
            )
            .drop("previous_month_revenue")
            .orderBy("month")
        )


def main() -> None:
    """Run sales analyses against the Parquet files generated by ``main.py``."""
    config.configure_logging()
    spark: SparkSession | None = None
    try:
        spark = SalesAnalytics.create_spark_session()
        orders = SalesAnalytics.load_parquet(config.RAW_DATA_DIR / "orders.parquet", spark)
        products = SalesAnalytics.load_parquet(config.RAW_DATA_DIR / "products.parquet", spark)

        results: tuple[tuple[str, DataFrame], ...] = (
            ("Top customers by revenue", SalesAnalytics.top_customers_by_revenue(orders, products)),
            ("Top customers by quantity", SalesAnalytics.top_customers_by_quantity(orders, n=20)),
            ("Sales by category", SalesAnalytics.sales_by_category(orders, products)),
            ("Monthly revenue trends", SalesAnalytics.monthly_trends(orders, products)),
        )
        for title, dataframe in results:
            LOGGER.info("%s", title)
            dataframe.show(20, truncate=False)
    except Exception:
        LOGGER.exception("Sales analytics failed")
        raise
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
