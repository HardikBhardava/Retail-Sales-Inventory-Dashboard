# ==========================================================
# Retail Sales & Inventory Analytics Dashboard
# app.py
# ==========================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path

# Page Configuration

st.set_page_config(
    page_title="Retail Sales & Inventory Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# Load Data
# ==========================================================

@st.cache_data
def load_data():
    data_path = Path("data/retail_sales_inventory.csv")

    if not data_path.exists():
        st.error("Dataset not found. Please run generate_data.py first.")
        st.stop()

    df = pd.read_csv(data_path, parse_dates=["date"])
    return df


df = load_data()

# Sidebar Filters

st.sidebar.title("Dashboard Filters")

# Date filter
min_date = df["date"].min().date()
max_date = df["date"].max().date()

selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Category filter
categories = sorted(df["category"].unique())

selected_categories = st.sidebar.multiselect(
    "Select Product Categories",
    categories,
    default=categories
)

# Store filter
locations = sorted(df["store_location"].unique())

selected_locations = st.sidebar.multiselect(
    "Select Store Locations",
    locations,
    default=locations
)

# Apply Filters

filtered_df = df.copy()

if len(selected_dates) == 2:
    start_date, end_date = selected_dates

    filtered_df = filtered_df[
        (filtered_df["date"].dt.date >= start_date) &
        (filtered_df["date"].dt.date <= end_date)
    ]

filtered_df = filtered_df[
    (filtered_df["category"].isin(selected_categories)) &
    (filtered_df["store_location"].isin(selected_locations))
]

# Dashboard Title

st.title("Retail Sales & Inventory Analytics Dashboard")

st.write(
    """
    Interactive retail analytics dashboard for:
    - Sales monitoring
    - Inventory analysis
    - Revenue tracking
    - Low-stock detection
    - Product performance analysis
    """
)

st.divider()

# KPI Metrics

total_revenue = filtered_df["revenue"].sum()

total_units_sold = filtered_df["units_sold"].sum()

total_orders = filtered_df["orders"].sum()

avg_order_value = total_revenue / max(total_orders, 1)

low_stock_items = filtered_df[
    filtered_df["stock_level"] <= filtered_df["reorder_point"]
]["product_id"].nunique()

# KPI layout
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Revenue",
    f"€{total_revenue:,.0f}"
)

col2.metric(
    "Units Sold",
    f"{total_units_sold:,.0f}"
)

col3.metric(
    "Average Order Value",
    f"€{avg_order_value:,.2f}"
)

col4.metric(
    "Low-Stock Products",
    f"{low_stock_items}"
)

st.divider()

# Revenue Trend

st.subheader("Revenue Trend Over Time")

revenue_trend = (
    filtered_df.groupby("date", as_index=False)["revenue"]
    .sum()
    .sort_values("date")
)

fig_revenue = px.line(
    revenue_trend,
    x="date",
    y="revenue",
    title="Daily Revenue Trend",
    markers=True
)

st.plotly_chart(
    fig_revenue,
    width="stretch"
)

# Revenue by Category and Store

left_col, right_col = st.columns(2)

# Revenue by category
category_sales = (
    filtered_df.groupby("category", as_index=False)["revenue"]
    .sum()
    .sort_values("revenue", ascending=False)
)

fig_category = px.bar(
    category_sales,
    x="category",
    y="revenue",
    title="Revenue by Product Category",
    text_auto=".2s"
)

left_col.plotly_chart(
    fig_category,
    width="stretch"
)

# Revenue by location
location_sales = (
    filtered_df.groupby("store_location", as_index=False)["revenue"]
    .sum()
    .sort_values("revenue", ascending=False)
)

fig_location = px.bar(
    location_sales,
    x="store_location",
    y="revenue",
    title="Revenue by Store Location",
    text_auto=".2s"
)

right_col.plotly_chart(
    fig_location,
    width="stretch"
)

# ==========================================================
# Top Products Table
# ==========================================================

st.subheader("Top-Selling Products")

top_products = (
    filtered_df.groupby(
        ["product_id", "product_name", "category"],
        as_index=False
    )
    .agg({
        "units_sold": "sum",
        "revenue": "sum"
    })
    .sort_values("revenue", ascending=False)
    .head(10)
)

st.dataframe(
    top_products,
    width="stretch"
)

# ==========================================================
# Inventory Risk Analysis
# ==========================================================

st.subheader("Inventory Risk Analysis")

inventory_summary = (
    filtered_df.groupby(
        ["product_id", "product_name", "category"],
        as_index=False
    )
    .agg({
        "stock_level": "mean",
        "reorder_point": "mean",
        "units_sold": "sum",
        "revenue": "sum"
    })
)

inventory_summary["stock_status"] = np.where(
    inventory_summary["stock_level"] <= inventory_summary["reorder_point"],
    "Low Stock",
    "Healthy Stock"
)

# Low stock table
low_stock_table = inventory_summary[
    inventory_summary["stock_status"] == "Low Stock"
].sort_values("stock_level")

st.dataframe(
    low_stock_table,
    width="stretch"
)

# Scatter plot
fig_inventory = px.scatter(
    inventory_summary,
    x="stock_level",
    y="units_sold",
    size="revenue",
    color="stock_status",
    hover_data=["product_name", "category"],
    title="Stock Level vs Units Sold"
)

st.plotly_chart(
    fig_inventory,
    width="stretch"
)

# ==========================================================
# Download Filtered Data
# ==========================================================

st.subheader("Download Filtered Dataset")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="filtered_retail_data.csv",
    mime="text/csv"
)

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "Portfolio Project | Retail Sales & Inventory Analytics Dashboard "
    "built with Streamlit, Plotly, Pandas, and Python."
)