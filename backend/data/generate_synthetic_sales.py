"""Create realistic-looking synthetic transaction files for the upload demo."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PRODUCTS = [
    ("Everyday Coffee", "Beverages", 8.5),
    ("Herbal Tea", "Beverages", 7.2),
    ("Granola Pack", "Grocery", 6.8),
    ("Gift Candle", "Home", 18.0),
    ("Cotton Tote", "Accessories", 14.5),
    ("Travel Mug", "Accessories", 22.0),
    ("Desk Notebook", "Stationery", 9.0),
    ("Wireless Charger", "Electronics", 29.0),
    ("Phone Stand", "Electronics", 17.5),
    ("Natural Soap Set", "Personal Care", 12.5),
]
REGIONS = ["North", "South", "East", "West", "Central"]
CHANNELS = ["Web", "Mobile app", "Store"]
PLANS = ["Essential", "Plus", "Premium"]


def build_file(name: str, seed: int, trend: float, seasonal: float, noise: float) -> None:
    rng = np.random.default_rng(seed)
    customers = [f"CUST-{i:05d}" for i in range(1, 321)]
    customer_region = rng.choice(REGIONS, len(customers), p=[.22, .18, .2, .2, .2])
    customer_channel = rng.choice(CHANNELS, len(customers), p=[.48, .32, .2])
    customer_age = rng.integers(21, 69, len(customers))
    customer_tenure = rng.integers(2, 49, len(customers))
    customer_plan = rng.choice(PLANS, len(customers), p=[.46, .36, .18])
    customer_auto_renew = rng.choice(["Yes", "No"], len(customers), p=[.72, .28])
    customer_satisfaction = np.clip(rng.normal(7.1, 1.45, len(customers)), 2, 10).round(1)
    customer_support = np.clip(rng.poisson(1.4, len(customers)), 0, 8)
    customer_loyalty = np.clip(
        0.35 + customer_tenure / 90 + (customer_plan == "Premium") * 0.18
        + (customer_auto_renew == "Yes") * 0.12 + rng.normal(0, 0.06, len(customers)),
        0.05,
        0.98,
    )
    customer_churn_score = (
        0.9 - customer_loyalty * 0.85
        + (customer_satisfaction < 6) * 0.22
        + (customer_support >= 4) * 0.16
        + (customer_auto_renew == "No") * 0.14
    )
    customer_churn = customer_churn_score > np.quantile(customer_churn_score, 0.7)

    months = pd.date_range("2024-01-01", "2025-06-01", freq="MS")
    rows: list[dict] = []
    for month_index, month in enumerate(months):
        month_factor = 1 + trend * month_index / max(1, len(months) - 1)
        seasonal_factor = 1 + seasonal * np.sin((month.month - 1) / 12 * 2 * np.pi)
        active_probability = np.clip(
            0.18 * month_factor * seasonal_factor * (1 + customer_loyalty * 0.55),
            0.04,
            0.75,
        )
        for customer_index, customer in enumerate(customers):
            if rng.random() > active_probability[customer_index]:
                continue
            transactions = 1 + int(rng.random() < 0.16)
            for _ in range(transactions):
                product, category, base_price = PRODUCTS[rng.integers(len(PRODUCTS))]
                quantity = int(np.clip(rng.poisson(2.0 + customer_loyalty[customer_index] * 2), 1, 10))
                price = base_price * rng.normal(1.0, 0.045)
                promotion = rng.random() < (0.12 + 0.08 * (month.month in (11, 12)))
                discount = rng.choice([0.0, 0.05, 0.10, 0.15], p=[.58, .2, .16, .06]) if promotion else 0.0
                days = int(rng.integers(0, max(1, (month + pd.offsets.MonthEnd(0) - month).days)))
                date = month + pd.Timedelta(days=days, hours=int(rng.integers(8, 22)))
                revenue = round(quantity * price * (1 - discount), 2)
                rows.append(
                    {
                        "CustomerID": customer,
                        "InvoiceDate": date.strftime("%Y-%m-%d %H:%M:%S"),
                        "Product": product,
                        "Category": category,
                        "Quantity": quantity,
                        "UnitPrice": round(price, 2),
                        "Discount": discount,
                        "Revenue": revenue,
                        "Channel": customer_channel[customer_index],
                        "Region": customer_region[customer_index],
                        "Promotion": "Yes" if promotion else "No",
                        "Age": int(customer_age[customer_index]),
                        "TenureMonths": int(customer_tenure[customer_index]),
                        "SubscriptionPlan": customer_plan[customer_index],
                        "AutoRenew": customer_auto_renew[customer_index],
                        "SatisfactionScore": float(customer_satisfaction[customer_index]),
                        "SupportTickets90d": int(customer_support[customer_index]),
                        "CustomerChurn": "Yes" if customer_churn[customer_index] else "No",
                    }
                )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["InvoiceDate", "CustomerID"]).reset_index(drop=True)
    frame.to_csv(ROOT / name, index=False)
    print(f"{name}: {len(frame):,} transactions, {frame['InvoiceDate'].str[:7].nunique()} months")


if __name__ == "__main__":
    build_file("synthetic_sales_steady.csv", seed=11, trend=0.10, seasonal=0.12, noise=0.08)
    build_file("synthetic_sales_seasonal.csv", seed=23, trend=0.05, seasonal=0.34, noise=0.10)
    build_file("synthetic_sales_declining.csv", seed=37, trend=-0.22, seasonal=0.18, noise=0.22)
