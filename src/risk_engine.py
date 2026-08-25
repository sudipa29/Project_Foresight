import pandas as pd
import os

PROCESSED_PATH = r"E:\Zidio_Development_Internship\Project_Foresight\data\processed"

sales = pd.read_csv(os.path.join(PROCESSED_PATH, "sales_clean.csv"))
customers = pd.read_csv(os.path.join(PROCESSED_PATH, "customers_clean.csv"))

print("Sales shape:", sales.shape)
print("Customers shape:", customers.shape)

print("\nSales columns:")
print(sales.columns.tolist())

print("\nCustomer columns:")
print(customers.columns.tolist())

# Check Customer ID Names
print("\nSales customer_id sample:")
print(sales["customer_id"].head(20))

print("\nCustomer ID sample:")
print(customers["customer_id"].head(20))

# Verify
customer_columns = [
    "age",
    "gender",
    "loyalty_segment",
    "preferred_channel",
    "registration_date"
]

print(sales_customer[customer_columns].isna().sum()
)

# Fix Store columns
sales_customer = sales_customer.rename(
    columns={
        "city_x": "store_city",
        "city_y": "customer_city"
    }
)

# ==========================================
# PROMOTION MAPPING
# ==========================================

promotions = pd.read_csv(
    os.path.join(PROCESSED_PATH, "promotions_clean.csv")
)

promotions["start_date"] = pd.to_datetime(
    promotions["start_date"],
    errors="coerce"
)

promotions["end_date"] = pd.to_datetime(
    promotions["end_date"],
    errors="coerce"
)

sales_customer["date"] = pd.to_datetime(
    sales_customer["date"],
    errors="coerce"
)

# Calculate promotion duration
promotions["duration_days"] = (
    promotions["end_date"]
    - promotions["start_date"]
).dt.days + 1

# Shorter/specific promotions get priority
promotions = promotions.sort_values(
    ["start_date", "duration_days"],
    ascending=[True, True]
).reset_index(drop=True)

# Create default promotion columns
sales_customer["promotion_flag"] = 0
sales_customer["promo_id"] = pd.NA
sales_customer["promo_name"] = "No Promotion"
sales_customer["promo_type"] = "No Promotion"
sales_customer["promo_discount_pct"] = 0.0

# Assign promotion
for _, promo in promotions.iterrows():

    mask = (
        sales_customer["date"].between(
            promo["start_date"],
            promo["end_date"]
        )
        &
        (sales_customer["promotion_flag"] == 0)
    )

    sales_customer.loc[
        mask,
        "promotion_flag"
    ] = 1

    sales_customer.loc[
        mask,
        "promo_id"
    ] = promo["promo_id"]

    sales_customer.loc[
        mask,
        "promo_name"
    ] = promo["promo_name"]

    sales_customer.loc[
        mask,
        "promo_type"
    ] = promo["promo_type"]

    sales_customer.loc[
        mask,
        "promo_discount_pct"
    ] = promo["discount_pct"]

print("\nPromotion mapping completed.")

print(
    sales_customer["promotion_flag"]
    .value_counts()
)

print("\nTop promotions:")
print(
    sales_customer["promo_name"]
    .value_counts()
    .head(15)
)

# Create a Separate Inventory Dataset
inventory = pd.read_csv(os.path.join(PROCESSED_PATH,"inventory_clean.csv"))
inventory["snapshot_date"] = pd.to_datetime(
    inventory["snapshot_date"]
)
print(inventory["snapshot_date"].value_counts())

# Daily Demand Dataset
daily_demand = (sales_customer.groupby(["date","store_id","sku_id"
        ],
        as_index=False
    )
    .agg(
        units_sold=("quantity", "sum"),
        revenue=("total_value", "sum"),
        avg_discount=("discount_pct", "mean"),
        promotion_flag=("promotion_flag", "max")
    )
)
print("Step 14 Complete: daily_demand created!")

sku_master = pd.read_csv(os.path.join(PROCESSED_PATH, "skus_clean.csv"))
print("Actual columns in skus_clean.csv:", sku_master.columns.tolist())

# Add SKU Information
sku_columns = [
    "sku_id",
    "sku_name",
    "category",
    "subcategory",
    "brand",
    "cost_price"
]
sku_master = sku_master[sku_columns].drop_duplicates("sku_id")

# Merge SKU Information
daily_demand = pd.merge(
    daily_demand,
    sku_master,
    on="sku_id",
    how="left",
    validate="many_to_one"
)
# Load directly from the clean Store file
store_master = pd.read_csv(os.path.join(PROCESSED_PATH, "stores_clean.csv"))
print("Actual columns in stores_clean.csv:", store_master.columns.tolist())

# If your store file has a column just named 'city', rename it to 'store_city' so it matches
if "city" in store_master.columns:
    store_master = store_master.rename(columns={"city": "store_city"})

store_columns = [
    "store_id", 
    "store_name", 
    "store_city", 
    "store_type"
]

store_master = store_master[store_columns].drop_duplicates("store_id")

# Merge Store Information
daily_demand = pd.merge(
    daily_demand,
    store_master,
    on="store_id",
    how="left",
    validate="many_to_one"
)

# Finally, calculate profit
daily_demand["profit"] = daily_demand["revenue"] - (daily_demand["units_sold"] * daily_demand["cost_price"])
print("Profit calculated successfully!")

sales = pd.read_csv("data/processed/sales_clean.csv")
customers = pd.read_csv("data/processed/customers_clean.csv")

if "cust_id" in customers.columns:
    customers = customers.rename(
        columns={"cust_id": "customer_id"}
    )

def standardize_id(series):
     return pd.to_numeric(
        series,
        errors="coerce"
    ).astype("Int64")

sales["customer_id"] = standardize_id(sales["customer_id"])

customers["customer_id"] = standardize_id(customers["customer_id"])

print(
    "Missing customer IDs:",
    sales["customer_id"].isna().sum()
)

print(
    "Known customer IDs:",
    sales["customer_id"].notna().sum()
)

sales_ids = set(
    sales.loc[
        sales["customer_id"] != "Unknown",
        "customer_id"
    ]
)

customer_ids = set(
    customers["customer_id"]
)

matched = sales_ids & customer_ids

print("Sales unique customer IDs:", len(sales_ids))
print("Customer master IDs:", len(customer_ids))
print("Matched IDs:", len(matched))

match_rate = (
    len(matched) / len(sales_ids) * 100
)

print(f"Customer Match Rate: {match_rate:.2f}%")

# Unmatched customer id
unmatched_ids = sales_ids - customer_ids

print("Unmatched customer IDs:")
print(unmatched_ids)

unmatched_sales = sales[
    sales["customer_id"].isin(unmatched_ids)
]

print(unmatched_sales)
print("Number of sales records:",len(unmatched_sales))

sales_customer = pd.merge(
    sales,
    customers,
    on="customer_id",
    how="left",
    validate="many_to_one"
)

# Check customer columns
customer_columns = [
    "age",
    "gender",
    "loyalty_segment",
    "preferred_channel",
    "registration_date"
]

print(
    sales_customer[customer_columns].isnull().sum()
)

sales_customer["customer_age_missing"] = (
    sales_customer["age"].isna().astype(int)
)

sales_customer["gender"] = (
    sales_customer["gender"]
    .fillna("Unknown")
)

sales_customer["loyalty_segment"] = (
    sales_customer["loyalty_segment"]
    .fillna("Unknown")
)

sales_customer["preferred_channel"] = (
    sales_customer["preferred_channel"]
    .fillna("Unknown")
)

sales_customer["registration_date"] = pd.to_datetime(
    sales_customer["registration_date"],
    errors="coerce"
)

# Check the result
print(
    sales_customer.shape
)

print(
    sales_customer[
        [
            "customer_id",
            "age",
            "gender",
            "loyalty_segment",
            "preferred_channel"
        ]
    ].head(10)
)

# Checking Inventory
inventory = pd.read_csv(
    "data/processed/inventory_clean.csv"
)

inventory["snapshot_date"] = pd.to_datetime(
    inventory["snapshot_date"],
    errors="coerce"
)

print("Inventory shape:")
print(inventory.shape)

print("\nSnapshot dates:")
print(
    inventory["snapshot_date"]
    .value_counts()
    .sort_index()
)
print("\nNumber of unique snapshot dates:")
print(
    inventory["snapshot_date"].nunique()
)

print("\nMinimum snapshot date:")
print(
    inventory["snapshot_date"].min()
)

print("\nMaximum snapshot date:")
print(
    inventory["snapshot_date"].max()
)

missing_customer_pct = (
    sales["customer_id"].isna().mean() * 100
)

print(
    f"Sales without customer ID: "
    f"{missing_customer_pct:.2f}%"
)
# Create promotion priority as they overlap
promotions["duration_days"] = (
    promotions["end_date"]
    - promotions["start_date"]
).dt.days + 1

promotions = promotions.sort_values(
    ["start_date", "duration_days"]
)