# ==========================================================
# Retail Sales & Inventory Analytics Dashboard
# app.py
# ==========================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# ==========================================================
# Business Insights
# ==========================================================

st.subheader("Business Insights")

if not filtered_df.empty:
    best_category = (
        filtered_df.groupby("category")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .idxmax()
    )

    best_location = (
        filtered_df.groupby("store_location")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .idxmax()
    )

    highest_units_product = (
        filtered_df.groupby("product_name")["units_sold"]
        .sum()
        .sort_values(ascending=False)
        .idxmax()
    )

    low_stock_count = low_stock_items

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.info(f"Highest revenue category: **{best_category}**")
        st.info(f"Best-performing store location: **{best_location}**")

    with insight_col2:
        st.warning(f"Top-selling product by units: **{highest_units_product}**")
        st.warning(f"Number of products at low-stock risk: **{low_stock_count}**")

else:
    st.warning("No data available for the selected filters.")

st.divider()

# ==========================================================
# Inventory Optimization KPIs
# ==========================================================

st.subheader("Inventory Optimization KPIs")

inventory_kpi = (
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

# Number of days in selected data
num_days = max(filtered_df["date"].nunique(), 1)

inventory_kpi["avg_daily_sales"] = inventory_kpi["units_sold"] / num_days

inventory_kpi["days_of_inventory_remaining"] = (
    inventory_kpi["stock_level"] / inventory_kpi["avg_daily_sales"].replace(0, np.nan)
)

inventory_kpi["inventory_turnover"] = (
    inventory_kpi["units_sold"] / inventory_kpi["stock_level"].replace(0, np.nan)
)

inventory_kpi["recommended_reorder_quantity"] = np.where(
    inventory_kpi["stock_level"] <= inventory_kpi["reorder_point"],
    inventory_kpi["reorder_point"] * 2 - inventory_kpi["stock_level"],
    0
)

inventory_kpi["days_of_inventory_remaining"] = inventory_kpi[
    "days_of_inventory_remaining"
].round(1)

inventory_kpi["inventory_turnover"] = inventory_kpi[
    "inventory_turnover"
].round(2)

inventory_kpi["recommended_reorder_quantity"] = inventory_kpi[
    "recommended_reorder_quantity"
].round(0)

risk_products = inventory_kpi[
    inventory_kpi["days_of_inventory_remaining"] < 7
]["product_id"].nunique()

avg_inventory_days = inventory_kpi["days_of_inventory_remaining"].mean()

total_reorder_quantity = inventory_kpi["recommended_reorder_quantity"].sum()

kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric(
    "Products with < 7 Days Stock",
    f"{risk_products}"
)

kpi2.metric(
    "Avg. Inventory Days Remaining",
    f"{avg_inventory_days:.1f} days"
)

kpi3.metric(
    "Recommended Reorder Quantity",
    f"{total_reorder_quantity:,.0f} units"
)

st.dataframe(
    inventory_kpi.sort_values("days_of_inventory_remaining"),
    width="stretch"
)

fig_inventory_days = px.bar(
    inventory_kpi.sort_values("days_of_inventory_remaining").head(10),
    x="product_name",
    y="days_of_inventory_remaining",
    color="category",
    title="Products with Lowest Inventory Coverage"
)

st.plotly_chart(
    fig_inventory_days,
    width="stretch"
)

st.divider()

# ==========================================================
# Advanced Demand Forecasting
# ==========================================================

st.subheader("Advanced Demand Forecasting")

forecast_df = (
    filtered_df.groupby("date", as_index=False)["units_sold"]
    .sum()
    .sort_values("date")
)

if len(forecast_df) >= 30:

    # ------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------

    forecast_df["day_number"] = np.arange(len(forecast_df))

    forecast_df["day_of_week"] = forecast_df["date"].dt.dayofweek

    forecast_df["month"] = forecast_df["date"].dt.month

    forecast_df["rolling_mean_7"] = (
        forecast_df["units_sold"]
        .rolling(window=7)
        .mean()
    )

    forecast_df["rolling_mean_14"] = (
        forecast_df["units_sold"]
        .rolling(window=14)
        .mean()
    )

    forecast_df = forecast_df.dropna()

    # ------------------------------------------------------
    # Features and target
    # ------------------------------------------------------

    features = [
        "day_number",
        "day_of_week",
        "month",
        "rolling_mean_7",
        "rolling_mean_14"
    ]

    X = forecast_df[features]

    y = forecast_df["units_sold"]

    # ------------------------------------------------------
    # Train-test split
    # ------------------------------------------------------

    split_index = int(len(forecast_df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # ------------------------------------------------------
    # Random Forest Model
    # ------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metric_col1, metric_col2 = st.columns(2)

    metric_col1.metric(
        "Forecast MAE",
        f"{mae:.2f} units"
    )

    metric_col2.metric(
        "Forecast RMSE",
        f"{rmse:.2f} units"
    )

    # ------------------------------------------------------
    # Historical predictions
    # ------------------------------------------------------

    forecast_df["prediction"] = model.predict(X)

    # ------------------------------------------------------
    # Future Forecast
    # ------------------------------------------------------

    future_days = 30

    last_day_number = forecast_df["day_number"].max()

    future_dates = pd.date_range(
        start=forecast_df["date"].max() + pd.Timedelta(days=1),
        periods=future_days
    )

    future_df = pd.DataFrame({
        "date": future_dates
    })

    future_df["day_number"] = np.arange(
        last_day_number + 1,
        last_day_number + future_days + 1
    )

    future_df["day_of_week"] = future_df["date"].dt.dayofweek

    future_df["month"] = future_df["date"].dt.month

    last_7_mean = forecast_df["units_sold"].tail(7).mean()

    last_14_mean = forecast_df["units_sold"].tail(14).mean()

    future_df["rolling_mean_7"] = last_7_mean

    future_df["rolling_mean_14"] = last_14_mean

    future_df["prediction"] = model.predict(
        future_df[features]
    )

    # ------------------------------------------------------
    # Combine historical + forecast
    # ------------------------------------------------------

    combined_df = pd.concat([
        forecast_df[["date", "units_sold", "prediction"]],
        future_df[["date", "prediction"]]
    ])

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    fig_forecast = px.line(
        combined_df,
        x="date",
        y=["units_sold", "prediction"],
        title="Actual vs Forecasted Demand"
    )

    st.plotly_chart(
        fig_forecast,
        width="stretch"
    )

    # ------------------------------------------------------
    # Future Forecast Table
    # ------------------------------------------------------

    st.write("30-Day Demand Forecast")

    forecast_table = future_df[
        ["date", "prediction"]
    ].rename(columns={
        "prediction": "forecasted_units_sold"
    })

    forecast_table["forecasted_units_sold"] = (
        forecast_table["forecasted_units_sold"]
        .round(0)
    )

    st.dataframe(
        forecast_table,
        width="stretch"
    )

    # ------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------

    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    fig_importance = px.bar(
        importance_df,
        x="feature",
        y="importance",
        title="Random Forest Feature Importance"
    )

    st.plotly_chart(
        fig_importance,
        width="stretch"
    )

else:
    st.warning(
        "Not enough data available for forecasting."
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