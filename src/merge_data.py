# ==========================================
# Phase 4 - Merge Datasets
# ==========================================

import pandas as pd
import os

# Path to raw data folder
DATA_PATH = r"E:\Zidio_Development_Internship\Project_Foresight\data\raw"
PROCESSED_PATH = r"E:\Zidio_Development_Internship\Project_Foresight\data\processed"

sales = pd.read_csv(os.path.join(PROCESSED_PATH, "sales_clean.csv"))
inventory = pd.read_csv(os.path.join(PROCESSED_PATH, "inventory_clean.csv"))
customers = pd.read_csv(os.path.join(PROCESSED_PATH, "customers_clean.csv"))
stores = pd.read_csv(os.path.join(PROCESSED_PATH, "stores_clean.csv"))
skus = pd.read_csv(os.path.join(PROCESSED_PATH, "skus_clean.csv"))
promotions = pd.read_csv(os.path.join(PROCESSED_PATH, "promotions_clean.csv"))

# Convert Date Columns to Datetime Format
sales["date"] = pd.to_datetime(sales["date"])
inventory["snapshot_date"] = pd.to_datetime(inventory["snapshot_date"])
inventory["last_restock_date"] = pd.to_datetime(inventory["last_restock_date"])
stores["opening_date"] = pd.to_datetime(stores["opening_date"])
customers["registration_date"] = pd.to_datetime(customers["registration_date"])
promotions["start_date"] = pd.to_datetime(promotions["start_date"])
promotions["end_date"] = pd.to_datetime(promotions["end_date"])

# Merge Sales with Store Data
# Force both customer_id columns to be strings so they can merge cleanly
sales["customer_id"] = sales["customer_id"].astype(str)
customers["customer_id"] = customers["customer_id"].astype(str)
# Now do the merge
sales_store = pd.merge(sales, stores, on="store_id", how="left")

# Merge Sales with Customer Data
sales_store_customer = pd.merge(sales_store, customers, on="customer_id", how="left")
# Merge Sales with SKU Data
skus_renamed = skus.rename(columns={"unit_price": "msrp_unit_price"})
sales_store_customer_sku = pd.merge(sales_store_customer, skus_renamed, on="sku_id", how="left")
# Merge Inventory
Merged_Data = pd.merge(sales_store_customer_sku,inventory,on=["store_id", "sku_id"],how="left")
# Rename unit_price to actual_unit_price
Merged_Data = Merged_Data.rename(columns={"unit_price": "actual_unit_price"})

# Promotion Information that labels promotion/ no promotion
Merged_Data["promotion_flag"] = 0
Merged_Data["promo_name"] = "No Promotion"

Merged_Data['date'] = pd.to_datetime(Merged_Data['date'])
promotions['start_date'] = pd.to_datetime(promotions['start_date'])
promotions['end_date'] = pd.to_datetime(promotions['end_date'])

for _, promo in promotions.iterrows():

    mask = (
        (Merged_Data["date"] >= promo["start_date"]) &
        (Merged_Data["date"] <= promo["end_date"]) &
        (Merged_Data["discount_pct"] == promo["discount_pct"])
    )

    Merged_Data.loc[mask, "promotion_flag"] = 1
    Merged_Data.loc[mask, "promo_name"] = promo["promo_name"]
print("Merged Merged_Data Dataset Shape:", Merged_Data.shape)

# Create Calendar Features
Merged_Data["Year"] = Merged_Data["date"].dt.year
Merged_Data["Month"] = Merged_Data["date"].dt.month
Merged_Data["Month_Name"] = Merged_Data["date"].dt.month_name()
Merged_Data["Week"] = Merged_Data["date"].dt.isocalendar().week
Merged_Data["Quarter"] = Merged_Data["date"].dt.quarter
Merged_Data["Day"] = Merged_Data["date"].dt.day
Merged_Data["Day_Name"] = Merged_Data["date"].dt.day_name()
Merged_Data["Weekend"] = Merged_Data["Day_Name"].isin(["Saturday", "Sunday"]).astype(int)

# Create season
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Autumn"

Merged_Data["Season"] = Merged_Data["Month"].apply(get_season)

# Calculate profit
print(Merged_Data.columns.tolist())
Merged_Data["profit"] = (Merged_Data["actual_unit_price"] - Merged_Data["cost_price"]) * Merged_Data["quantity"]

# Calculate profit margin
Merged_Data["Profit_Margin"] = (Merged_Data["profit"] / Merged_Data["total_value"]) * 100

# Final Dataset check
print(Merged_Data.shape)
print(Merged_Data.head())
print(Merged_Data.info())

# Save the merged dataset to a CSV file
Merged_Data.to_csv(os.path.join(PROCESSED_PATH, "merged_data.csv"), index=False)