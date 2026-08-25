# ==========================================
# Project FORESIGHT
# Phase 3 - Clean Data
# ==========================================

import pandas as pd
import os

# Path to raw data folder
DATA_PATH = r"E:\Zidio_Development_Internship\Project_Foresight\data\raw"
PROCESSED_PATH = r"E:\Zidio_Development_Internship\Project_Foresight\data\processed"

# Load datasets
sales = pd.read_csv(os.path.join(DATA_PATH, "bm_sales.csv"), low_memory=False)
inventory = pd.read_csv(os.path.join(DATA_PATH, "bm_inventory.csv"), low_memory=False)
customers = pd.read_csv(os.path.join(DATA_PATH, "bm_customers.csv"), low_memory=False)
stores = pd.read_csv(os.path.join(DATA_PATH, "bm_stores.csv"), low_memory=False)
skus = pd.read_csv(os.path.join(DATA_PATH, "bm_skus.csv"), low_memory=False)
promotions = pd.read_csv(os.path.join(DATA_PATH, "bm_promotions.csv"), low_memory=False)

# Create a Reusable Cleaning Function
def clean_dataset(df, dataset_name):

    print("="*60)
    print(f"Cleaning Dataset: {dataset_name}")
    print("="*60)

    print("Initial Shape:", df.shape)

    # Remove duplicate rows
    duplicates = df.duplicated().sum()
    print("Duplicate Rows:", duplicates)
    df = df.drop_duplicates()

    # Missing Values
    print("\nMissing Values:")
    print(df.isnull().sum())

    return df

# Clean duplicates in each dataset
sales=sales.drop_duplicates()
inventory=inventory.drop_duplicates()
customers=customers.drop_duplicates()
stores=stores.drop_duplicates()
skus=skus.drop_duplicates()
promotions=promotions.drop_duplicates()

# Handle Missing Values
# As only 1.5% of the customer_id values are missing, we can fill them with a placeholder value like "Unknown" or "Missing". This will allow us to retain those records for analysis without losing any data.
sales["customer_id"] = sales["customer_id"].fillna("Unknown")
print("Missing Values in Sales after filling customer_id:", sales["customer_id"].isnull().sum())

# Convert Date Columns to Datetime Format
sales["date"] = pd.to_datetime(sales["date"])
inventory["snapshot_date"] = pd.to_datetime(inventory["snapshot_date"])
inventory["last_restock_date"] = pd.to_datetime(inventory["last_restock_date"])
stores["opening_date"] = pd.to_datetime(stores["opening_date"])
customers["registration_date"] = pd.to_datetime(customers["registration_date"])
promotions["start_date"] = pd.to_datetime(promotions["start_date"])
promotions["end_date"] = pd.to_datetime(promotions["end_date"])

#show after conversion
print("Sales Date Column Type:", sales["date"].dtype)
print("Inventory Snapshot Date Column Type:", inventory["snapshot_date"].dtype)
print("Inventory Last Restock Date Column Type:", inventory["last_restock_date"].dtype)
print("Stores Opening Date Column Type:", stores["opening_date"].dtype)
print("Customers Registration Date Column Type:", customers["registration_date"].dtype)
print("Promotions Start Date Column Type:", promotions["start_date"].dtype)
print("Promotions End Date Column Type:", promotions["end_date"].dtype) 

# Fix Column Names
customers.rename(
    columns={"cust_id": "customer_id"},
    inplace=True
)
print("Customers Columns after renaming:", customers.columns)

# Check for negative quantities
# No negative quantities present in the dataset
print(sales[sales["quantity"] < 0])
print("Sales with non-negative quantities:", sales[sales["quantity"] >= 0].shape[0])
print("Sales with non-negative unit prices:", sales[sales["unit_price"] >= 0].shape[0])
print("Sales with non-negative total values:", sales[sales["total_value"] >= 0].shape[0])

# Validate Inventory Data
print("Inventory with non-negative stock on hand:", inventory[inventory["stock_on_hand"] >= 0].shape[0])
print("Inventory with non-negative reorder point:", inventory[inventory["reorder_point"] >= 0].shape[0])
print("Inventory with non-negative safety stock:", inventory[inventory["safety_stock"] >= 0].shape[0])

# Standardize Text
skus["category"] = (skus["category"].str.strip().str.title())
skus["subcategory"] = (skus["subcategory"].str.strip().str.title())
stores["city"] = (stores["city"].str.strip().str.title())
customers["gender"] = (customers["gender"].str.strip().str.title())
print("SKUs Category after standardization:", skus["category"].unique())
print("SKUs Subcategory after standardization:", skus["subcategory"].unique())
print("Stores City after standardization:", stores["city"].unique())
print("Customers Gender after standardization:", customers["gender"].unique())

# Verify Cost Price should not exceed Selling Price
invalid_price = skus[skus["cost_price"] > skus["unit_price"]]
print("SKUs with invalid pricing:", invalid_price.shape[0])

# Check for invalid discount percentages
print("Sales with invalid discount percentages:", sales[(sales["discount_pct"] < 0) | (sales["discount_pct"] > 100)].shape[0])

# Create the processed folder if it doesn't exist
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Save Cleaned Datasets
sales.to_csv(os.path.join(PROCESSED_PATH, "sales_clean.csv"),index=False)
inventory.to_csv(os.path.join(PROCESSED_PATH, "inventory_clean.csv"), index=False)
customers.to_csv(os.path.join(PROCESSED_PATH, "customers_clean.csv"), index=False)
stores.to_csv(os.path.join(PROCESSED_PATH, "stores_clean.csv"), index=False)
skus.to_csv(os.path.join(PROCESSED_PATH, "skus_clean.csv"), index=False)
promotions.to_csv(os.path.join(PROCESSED_PATH, "promotions_clean.csv"), index=False)

# Print Final Summary of Cleaned Datasets
print("="*60)
print("Data Cleaning Completed Successfully!")
print("="*60)
print("Sales:", sales.shape)
print("Inventory:", inventory.shape)
print("Customers:", customers.shape)
print("Stores:", stores.shape)
print("SKUs:", skus.shape)
print("Promotions:", promotions.shape)