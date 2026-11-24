"""Generate synthetic e-commerce data as pandas DataFrames."""

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker
from tqdm.auto import tqdm

if __package__:
    from src.config import CUSTOMERS_FILE, ORDERS_FILE, PRODUCTS_FILE, configure_logging
else:
    # Support IDEs that launch this file directly with ``python src/data_generator.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import CUSTOMERS_FILE, ORDERS_FILE, PRODUCTS_FILE, configure_logging

LOGGER = logging.getLogger(__name__)
CATEGORIES: tuple[str, ...] = ("Electronics", "Clothing", "Home", "Sports", "Books")


class SyntheticDataGenerator:
    """Create reproducible, fake e-commerce customers, products, and orders.

    The defaults generate 100,000 customers, 10,000 products, and 1,000,000
    orders. Order allocation follows an 80/20 Pareto pattern: a random 20% of
    customers receive 80% of orders, with Pareto-distributed weights within
    both customer groups.
    """

    def __init__(
        self,
        num_customers: int = 100_000,
        num_products: int = 10_000,
        num_orders: int = 1_000_000,
        seed: int = 42,
    ) -> None:
        """Initialize dataset sizes and seeded random number generators."""
        if min(num_customers, num_products, num_orders) < 1:
            raise ValueError("Customer, product, and order counts must be positive")
        if num_customers < 5:
            raise ValueError("num_customers must be at least 5 for the 80/20 order split")

        self.num_customers = num_customers
        self.num_products = num_products
        self.num_orders = num_orders
        self.seed = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.fake: Faker = Faker()
        self.fake.seed_instance(seed)
        LOGGER.info(
            "Initialized generator: customers=%d products=%d orders=%d seed=%d",
            num_customers,
            num_products,
            num_orders,
            seed,
        )

    def generate_customers(self, count: int | None = None) -> pd.DataFrame:
        """Generate customers with normally distributed ages centered near 35."""
        customer_count = count if count is not None else self.num_customers
        if customer_count < 1:
            raise ValueError("Customer count must be positive")

        LOGGER.info("Generating %d customers", customer_count)
        ages = np.clip(self.rng.normal(loc=35, scale=10, size=customer_count), 18, 90)
        now = pd.Timestamp.now().normalize()
        earliest = now - pd.Timedelta(days=365 * 5)
        customers: list[dict[str, Any]] = []
        for index in tqdm(range(customer_count), desc="Generating customers", unit="customer"):
            customers.append(
                {
                    "customer_id": f"CUST-{index + 1:07d}",
                    "name": self.fake.name(),
                    "email": self.fake.unique.email(),
                    "age": int(round(float(ages[index]))),
                    "city": self.fake.city(),
                    "country": self.fake.country(),
                    "registration_date": self.fake.date_between_dates(
                        date_start=earliest.date(), date_end=now.date()
                    ),
                }
            )
        result = pd.DataFrame.from_records(
            customers,
            columns=["customer_id", "name", "email", "age", "city", "country", "registration_date"],
        )
        LOGGER.info("Generated customer DataFrame with %d rows", len(result))
        return result

    def generate_products(self, count: int | None = None) -> pd.DataFrame:
        """Generate products with a category, price, stock level, and rating."""
        product_count = count if count is not None else self.num_products
        if product_count < 1:
            raise ValueError("Product count must be positive")

        LOGGER.info("Generating %d products", product_count)
        products: list[dict[str, Any]] = []
        for index in tqdm(range(product_count), desc="Generating products", unit="product"):
            products.append(
                {
                    "product_id": f"PROD-{index + 1:06d}",
                    "name": f"{self.fake.word().title()} {self.fake.word().title()} {index + 1}",
                    "category": str(self.rng.choice(CATEGORIES)),
                    "price": round(float(self.rng.uniform(10, 500)), 2),
                    "stock": int(self.rng.integers(0, 1_001)),
                    "rating": round(float(self.rng.uniform(1, 5)), 1),
                }
            )
        result = pd.DataFrame.from_records(
            products, columns=["product_id", "name", "category", "price", "stock", "rating"]
        )
        LOGGER.info("Generated product DataFrame with %d rows", len(result))
        return result

    def generate_orders(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        count: int | None = None,
    ) -> pd.DataFrame:
        """Generate orders, allocating 80% to a Pareto-weighted 20% of customers.

        Args:
            customers: Customer DataFrame containing ``customer_id``.
            products: Product DataFrame containing ``product_id``.
            count: Number of orders, defaulting to the configured order count.
        """
        order_count = count if count is not None else self.num_orders
        if order_count < 1:
            raise ValueError("Order count must be positive")
        if customers.empty or products.empty:
            raise ValueError("Customer and product DataFrames must not be empty")
        if "customer_id" not in customers or "product_id" not in products:
            raise ValueError("Input DataFrames must contain customer_id and product_id columns")

        LOGGER.info("Generating %d orders with an 80/20 customer allocation", order_count)
        customer_ids = customers["customer_id"].to_numpy()
        product_ids = products["product_id"].to_numpy()
        top_count = max(1, int(round(len(customer_ids) * 0.20)))
        customer_order = self.rng.permutation(len(customer_ids))
        top_customers = customer_order[:top_count]
        other_customers = customer_order[top_count:]

        top_orders = int(round(order_count * 0.80))
        group_labels = np.concatenate(
            (np.zeros(top_orders, dtype=np.int8), np.ones(order_count - top_orders, dtype=np.int8))
        )
        self.rng.shuffle(group_labels)
        customer_indices = np.empty(order_count, dtype=np.int64)
        for group, indices in ((0, top_customers), (1, other_customers)):
            order_positions = np.flatnonzero(group_labels == group)
            if not len(order_positions):
                continue
            if not len(indices):
                indices = top_customers
            weights = self.rng.pareto(a=1.16, size=len(indices)) + 1.0
            probabilities = weights / weights.sum()
            customer_indices[order_positions] = self.rng.choice(
                indices, size=len(order_positions), p=probabilities
            )

        start = np.datetime64("today", "D") - np.timedelta64(2 * 365, "D")
        day_offsets = self.rng.integers(0, 2 * 365, size=order_count)
        order_dates = start + day_offsets.astype("timedelta64[D]")
        product_indices = self.rng.integers(0, len(product_ids), size=order_count)
        quantities = self.rng.integers(1, 11, size=order_count)

        chunks: list[pd.DataFrame] = []
        chunk_size = 100_000
        for start_index in tqdm(
            range(0, order_count, chunk_size), desc="Generating orders", unit="chunk"
        ):
            end_index = min(start_index + chunk_size, order_count)
            positions = np.arange(start_index, end_index)
            chunks.append(
                pd.DataFrame(
                    {
                        "order_id": [f"ORD-{index + 1:09d}" for index in positions],
                        "customer_id": customer_ids[customer_indices[start_index:end_index]],
                        "product_id": product_ids[product_indices[start_index:end_index]],
                        "quantity": quantities[start_index:end_index],
                        "order_date": pd.to_datetime(order_dates[start_index:end_index]),
                    }
                )
            )
        result = pd.concat(chunks, ignore_index=True)
        LOGGER.info("Generated orders DataFrame with %d rows", len(result))
        return result

    def generate_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate and return customers, products, and orders in that order."""
        customers = self.generate_customers()
        products = self.generate_products()
        orders = self.generate_orders(customers, products)
        return customers, products, orders


def generate_data(
    customer_count: int = 100_000,
    product_count: int = 10_000,
    order_count: int = 1_000_000,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    """Generate default-sized datasets and save each DataFrame as a CSV file."""
    generator = SyntheticDataGenerator(customer_count, product_count, order_count, seed)
    customers, products, orders = generator.generate_all()
    for dataframe, path in (
        (customers, CUSTOMERS_FILE),
        (products, PRODUCTS_FILE),
        (orders, ORDERS_FILE),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)
        LOGGER.info("Saved %d rows to %s", len(dataframe), path)
    return CUSTOMERS_FILE, PRODUCTS_FILE, ORDERS_FILE


def main() -> None:
    """Generate the default datasets from the command line."""
    configure_logging()
    generate_data()


if __name__ == "__main__":
    main()
