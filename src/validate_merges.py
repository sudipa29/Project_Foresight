import pandas as pd
import os

PROCESSED_PATH = r"E:\Zidio_Development_Internship\Project_Foresight\data\processed"

# ==========================================
# 1. LOAD ALL DATA ONCE
# ==========================================
print("Loading datasets...")
sales = pd.read_csv(os.path.join(PROCESSED_PATH, "sales_clean.csv"))
customers = pd.read_csv(os.path.join(PROCESSED_PATH, "customers_clean.csv"))
promotions = pd.read_csv(os.path.join(PROCESSED_PATH, "promotions_clean.csv"))
inventory = pd.read_csv(os.path.join(PROCESSED_PATH, "inventory_clean.csv"))
sku_master = pd.read_csv(os.path.join(PROCESSED_PATH, "skus_clean.csv"))
store_master = pd.read_csv(os.path.join(PROCESSED_PATH, "stores_clean.csv"))

# ==========================================
# 2. STANDARDIZE & MERGE CUSTOMERS
# ==========================================
def standardize_id(series):
     return pd.to_numeric(series, errors="coerce").astype("Int64")

sales["customer_id"] = standardize_id(sales["customer_id"])
customers["customer_id"] = standardize_id(customers["customer_id"])

if "cust_id" in customers.columns:
    customers = customers.rename(columns={"cust_id": "customer_id"})

sales_customer = pd.merge(
    sales,
    customers,
    on="customer_id",
    how="left",
    validate="many_to_one"
)

# Handle anonymous/missing customers correctly
sales_customer["customer_age_missing"] = sales_customer["age"].isna().astype(int)
sales_customer["gender"] = sales_customer["gender"].fillna("Unknown")
sales_customer["loyalty_segment"] = sales_customer["loyalty_segment"].fillna("Unknown")
sales_customer["preferred_channel"] = sales_customer["preferred_channel"].fillna("Unknown")
sales_customer["registration_date"] = pd.to_datetime(sales_customer["registration_date"], errors="coerce")

print(f"Customer merge complete. Shape: {sales_customer.shape}")

# ==========================================
# 3. PROMOTION MAPPING
# ==========================================
promotions["start_date"] = pd.to_datetime(promotions["start_date"], errors="coerce")
promotions["end_date"] = pd.to_datetime(promotions["end_date"], errors="coerce")
sales_customer["date"] = pd.to_datetime(sales_customer["date"], errors="coerce")

# Calculate promotion duration & sort so shorter/specific promos get priority
promotions["duration_days"] = (promotions["end_date"] - promotions["start_date"]).dt.days + 1
promotions = promotions.sort_values(["start_date", "duration_days"], ascending=[True, True]).reset_index(drop=True)

# Create default promotion columns
sales_customer["promotion_flag"] = 0
sales_customer["promo_id"] = pd.NA
sales_customer["promo_name"] = "No Promotion"
sales_customer["promo_type"] = "No Promotion"
sales_customer["promo_discount_pct"] = 0.0

# Assign promotion (overlap priority handled by sorting)
for _, promo in promotions.iterrows():
    mask = (
        sales_customer["date"].between(promo["start_date"], promo["end_date"]) &
        (sales_customer["promotion_flag"] == 0)
    )
    sales_customer.loc[mask, "promotion_flag"] = 1
    sales_customer.loc[mask, "promo_id"] = promo["promo_id"]
    sales_customer.loc[mask, "promo_name"] = promo["promo_name"]
    sales_customer.loc[mask, "promo_type"] = promo["promo_type"]
    sales_customer.loc[mask, "promo_discount_pct"] = promo["discount_pct"]

print("Promotion mapping completed.")

print("\nPromotion Summary:")
print(
    sales_customer["promotion_flag"]
    .value_counts()
)
print("\nTop Promotions:")
print(
    sales_customer[
        sales_customer["promotion_flag"] == 1
    ]["promo_name"]
    .value_counts()
    .head(15)
)
# ==========================================
# 4. CREATE DAILY DEMAND DATASET
# ==========================================
daily_demand = (
    sales_customer.groupby(["date", "store_id", "sku_id"], as_index=False)
    .agg(
        units_sold=("quantity", "sum"),
        revenue=("total_value", "sum"),
        avg_discount=("discount_pct", "mean"),
        promotion_flag=("promotion_flag", "max"),
        promo_name=("promo_name", "first"), # Added based on ChatGPT's suggestion
        promo_type=("promo_type", "first")  # Added based on ChatGPT's suggestion
    )
)

print("Daily demand aggregated.")

# ==========================================
# 5. MERGE SKU & STORE INFO INTO DAILY DEMAND
# ==========================================
# SKU Merge
sku_columns = ["sku_id", "sku_name", "category", "subcategory", "brand", "cost_price"]
sku_master = sku_master[sku_columns].drop_duplicates("sku_id")

daily_demand = pd.merge(daily_demand, sku_master, on="sku_id", how="left", validate="many_to_one")

# Store Merge
if "city" in store_master.columns:
    store_master = store_master.rename(columns={"city": "store_city"})

store_columns = ["store_id", "store_name", "store_city", "store_type"]
store_master = store_master[store_columns].drop_duplicates("store_id")

daily_demand = pd.merge(daily_demand, store_master, on="store_id", how="left", validate="many_to_one")

# Calculate Gross Profit
daily_demand["gross_profit"] = daily_demand["revenue"] - (daily_demand["units_sold"] * daily_demand["cost_price"])

print("Daily Demand dataset finalized!")

# SKU Validation
print("\nSKU Merge Validation:")

print(
    "Missing SKU names:",
    daily_demand["sku_name"].isna().sum()
)

print(
    "Missing categories:",
    daily_demand["category"].isna().sum()
)

print("\nStore Merge Validation:")

print(
    "Missing store names:",
    daily_demand["store_name"].isna().sum()
)

print(
    "Missing store cities:",
    daily_demand["store_city"].isna().sum()
)

# Final Daily Demand
print("\nDaily Demand Shape:")
print(daily_demand.shape)

print("\nDaily Demand Columns:")
print(daily_demand.columns.tolist())

print("\nMissing Values:")
print(
    daily_demand.isna().sum()
)

# Duplicate
duplicate_count = daily_demand.duplicated(
    subset=["date", "store_id", "sku_id"]
).sum()

print(
    "\nDuplicate Date-Store-SKU combinations:",
    duplicate_count
)

# Check numerical sanity
print("\nNumerical Summary:")
print(
    daily_demand[
        [
            "units_sold",
            "revenue",
            "avg_discount",
            "cost_price",
            "gross_profit"
        ]
    ].describe()
)

print(
    "\nNegative units sold:",
    (daily_demand["units_sold"] < 0).sum()
)

print(
    "Negative revenue:",
    (daily_demand["revenue"] < 0).sum()
)

print(
    "Negative cost price:",
    (daily_demand["cost_price"] < 0).sum()
)

# Check the data range
print("\nDate Range:")

print(
    "Start:",
    daily_demand["date"].min()
)

print(
    "End:",
    daily_demand["date"].max()
)

print(
    "Number of days:",
    daily_demand["date"].nunique()
)

# Validate inventory separately

print("\nInventory Summary:")
print(inventory.shape)

print(
    inventory[
        [
            "stock_on_hand",
            "reorder_point",
            "safety_stock"
        ]
    ].describe()
)

print(
    "\nInventory snapshot:",
    inventory["snapshot_date"].min()
)

print(
    "Unique snapshot dates:",
    inventory["snapshot_date"].nunique()
)

# Duplicate validation
print(
    "\nDuplicate Date-Store-SKU combinations:",
    daily_demand.duplicated(
        subset=["date", "store_id", "sku_id"]
    ).sum()
)

# Negative values
print("\nNegative Value Checks:")
print("Negative units:", (daily_demand["units_sold"] < 0).sum())
print("Negative revenue:", (daily_demand["revenue"] < 0).sum())
print("Negative cost:", (daily_demand["cost_price"] < 0).sum())

# Save official analytics dataset
daily_demand.to_csv(
    os.path.join(
        PROCESSED_PATH,
        "analytics_dataset.csv"
    ),
    index=False
)

print("\nAnalytics dataset saved successfully.")
# ==========================================
# 6. SAVE FINAL FILES (Ready for EDA)
# ==========================================
sales_customer.to_csv(os.path.join(PROCESSED_PATH, "sales_customer_enriched.csv"), index=False)
daily_demand.to_csv(os.path.join(PROCESSED_PATH, "daily_demand.csv"), index=False)

# Inventory is a current snapshot, save it separately for later
inventory["snapshot_date"] = pd.to_datetime(inventory["snapshot_date"], errors="coerce")
inventory.to_csv(os.path.join(PROCESSED_PATH, "inventory_current.csv"), index=False)

print("\nSUCCESS: All files processed and saved correctly! Ready for Phase 5 (EDA).")