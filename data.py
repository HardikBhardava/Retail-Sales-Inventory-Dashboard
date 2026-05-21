import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

Path("data").mkdir(exist_ok=True)

num_days = 180
products = [
    ("P001", "Basic T-Shirt", "Clothing"),
    ("P002", "Slim Jeans", "Clothing"),
    ("P003", "Summer Dress", "Clothing"),
    ("P004", "Sneakers", "Shoes"),
    ("P005", "Leather Boots", "Shoes"),
    ("P006", "Handbag", "Accessories"),
    ("P007", "Belt", "Accessories"),
    ("P008", "Winter Jacket", "Outerwear"),
    ("P009", "Blazer", "Outerwear"),
    ("P010", "Sports Hoodie", "Sportswear"),
]

locations = ["Düsseldorf", "Essen", "Köln", "München", "Berlin"]

dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=num_days)

rows = []

for date in dates:
    month = date.month
    seasonal_factor = 1.0

    if month in [11, 12, 1]:
        seasonal_factor = 1.25
    elif month in [6, 7, 8]:
        seasonal_factor = 1.15

    for product_id, product_name, category in products:
        for location in locations:
            base_demand = np.random.randint(5, 30)
            units_sold = max(0, int(np.random.normal(base_demand * seasonal_factor, 5)))
            price = {
                "Clothing": np.random.uniform(20, 80),
                "Shoes": np.random.uniform(50, 150),
                "Accessories": np.random.uniform(15, 100),
                "Outerwear": np.random.uniform(80, 220),
                "Sportswear": np.random.uniform(30, 120),
                }[category]

            orders = max(1, int(units_sold / np.random.uniform(1.1, 2.5)))
            stock_level = max(0, int(np.random.normal(120 - units_sold * 1.5, 25)))
            reorder_point = np.random.randint(25, 60)
            revenue = units_sold * price

            rows.append({
                "date": date,
                "product_id": product_id,
                "product_name": product_name,
                "category": category,
                "store_location": location,
                "units_sold": units_sold,
                 "orders": orders,
                "price": round(price, 2),
                "revenue": round(revenue, 2),
                "stock_level": stock_level,
                "reorder_point": reorder_point
            })

df = pd.DataFrame(rows)
# df.to_csv("data/retail_sales_inventory.csv", index=False)
# print("Synthetic dataset created: data/retail_sales_inventory.csv")
print(df.shape)