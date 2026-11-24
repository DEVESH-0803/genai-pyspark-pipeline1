# E-commerce Data Pipeline

A small Python project that generates synthetic customers, products, and orders, then uses PySpark to produce business summaries. It is intended for practicing data engineering and analytics without using real customer data.

## Project structure

```text
genai-pyspark-pipeline1/
├── data/
│   ├── raw/                 # Generated CSV inputs
│   └── processed/           # Spark result CSV directories
├── notebooks/               # Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generator.py
│   └── spark_analytics.py
├── tests/                   # Test scripts
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or newer is recommended. Java 8, 11, or 17 must be installed and available to PySpark (set `JAVA_HOME` if needed).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the pipeline

From the project root, generate the default Parquet dataset and run the Spark analyses:

```bash
python main.py
python run_analytics.py
```

The generator writes `customers.parquet`, `products.parquet`, and `orders.parquet` under `data/raw/`, then reports elapsed time and file sizes. The Spark job prints top customers by revenue and quantity, revenue and units sold by category, and monthly revenue growth.

Change the defaults in `src/config.py` or call `generate_data(...)` with different row counts and a seed. Generated data is reproducible for a given seed. No real personal or payment data is collected; emails and addresses are synthetic.

## Analyses

- Revenue, order count, and units sold by product category
- Revenue and order count by month
- Top products ranked by revenue

All application functions have type hints and docstrings. Logging reports progress and errors to the console.
