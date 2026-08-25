# ==========================================
# Project FORESIGHT
# Phase 2 - Load Data
# ==========================================

import pandas as pd
import os

# Path to raw data folder
DATA_PATH = r"E:\Zidio_Development_Internship\Project_Foresight\data\raw"

# Load datasets
sales = pd.read_csv(os.path.join(DATA_PATH, "bm_sales.csv"))
inventory = pd.read_csv(os.path.join(DATA_PATH, "bm_inventory.csv"))
customers = pd.read_csv(os.path.join(DATA_PATH, "bm_customers.csv"))
stores = pd.read_csv(os.path.join(DATA_PATH, "bm_stores.csv"))
skus = pd.read_csv(os.path.join(DATA_PATH, "bm_skus.csv"))
promotions = pd.read_csv(os.path.join(DATA_PATH, "bm_promotions.csv"))

# Check files loaded successfully/not
# Shapes of the datasets(rows, columns)
print("Datasets Loaded Successfully!")

print("Sales Shape:", sales.shape)
print("Inventory Shape:", inventory.shape)
print("Customers Shape:", customers.shape)
print("Stores Shape:", stores.shape)
print("SKUs Shape:", skus.shape)
print("Promotions Shape:", promotions.shape)

# Verify the first few rows of each dataset
# Head provides a quick look at the first 5 rows of the dataset
# Tail provides a quick look at the last 5 rows of the dataset
print(sales.head())
print(inventory.head())
print(customers.head())
print(stores.head())
print(skus.head())
print(promotions.head())

print(sales.tail())

# Check dataset information and data types
# Info provides no.of rows, columns, data types, missing valuesand memory usage of the dataset
print(sales.info())
print(inventory.info())
print(customers.info())
print(stores.info())
print(skus.info())
print(promotions.info())

# Check for missing values in each dataset 
# sum().sum() gives the total number of missing values in the dataset
print("================================")
print("Missing values in Sales:", sales.isnull().sum().sum())
print("Missing values in Inventory:", inventory.isnull().sum().sum())
print("Missing values in Customers:", customers.isnull().sum().sum())
print("Missing values in Stores:", stores.isnull().sum().sum())
print("Missing values in SKUs:", skus.isnull().sum().sum())
print("Missing values in Promotions:", promotions.isnull().sum().sum())

# Check for duplicates in each dataset
# duplicated() returns a boolean Series denoting duplicate rows. 
# sum() counts the number of True values, which indicates the number of duplicate rows in the dataset.
print("================================")
print("Duplicates in Sales:", sales.duplicated().sum())
print("Duplicates in Inventory:", inventory.duplicated().sum())
print("Duplicates in Customers:", customers.duplicated().sum())
print("Duplicates in Stores:", stores.duplicated().sum())
print("Duplicates in SKUs:", skus.duplicated().sum())
print("Duplicates in Promotions:", promotions.duplicated().sum())

# check datatypes
print("============Data Types====================")
print("Data Types in Sales:", sales.dtypes)
print("Data Types in Inventory:", inventory.dtypes)
print("Data Types in Customers:", customers.dtypes)
print("Data Types in Stores:", stores.dtypes)
print("Data Types in SKUs:", skus.dtypes)
print("Data Types in Promotions:", promotions.dtypes)

# Generate summary statistics for each dataset
# it helps identify issues such as: Negative quantity, Very high prices, Zero sales
print("================================")
print("Summary Statistics for Sales:",sales.describe())
print("Summary Statistics for Inventory:",inventory.describe())
print("Summary Statistics for Customers:",customers.describe())
print("Summary Statistics for Stores:",stores.describe())
print("Summary Statistics for SKUs:",skus.describe())
print("Summary Statistics for Promotions:",promotions.describe())

# checking for unique values in each dataset
print("================================")
print(sales["channel"].value_counts())
print(promotions["promo_type"].value_counts())
print(customers["loyalty_segment"].value_counts())
print(customers["preferred_channel"].value_counts())
print(customers["city"].value_counts())
print(customers["gender"].value_counts())
print(skus["brand"].value_counts())
print(skus["category"].value_counts())
print(skus["subcategory"].value_counts())
print(stores["city"].value_counts())
print(stores["store_type"].value_counts())
