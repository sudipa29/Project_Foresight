# 📊 Project FORESIGHT

## Demand Intelligence, Forecasting & Inventory Optimization Platform

Project FORESIGHT is an end-to-end retail analytics and machine learning platform designed to analyze historical sales demand, identify demand patterns and drivers, forecast future demand, evaluate inventory risk, and generate actionable business recommendations.

The project combines:

- Data Analytics
- Statistical Analysis
- Exploratory Data Analysis
- Demand Classification
- Intermittent Demand Forecasting
- Machine Learning
- Time-Series Forecasting
- Inventory Intelligence
- Business Analytics
- Interactive Data Visualization

The final insights are presented through an interactive **Streamlit dashboard** designed for business decision-making.

---

# 🚀 Project Overview

Retail businesses frequently face two major inventory challenges:

### Overstock

Excess inventory increases:

- Holding costs
- Working capital requirements
- Storage costs
- Inventory aging
- Markdown risk

### Stockouts

Insufficient inventory can result in:

- Lost sales
- Poor customer experience
- Revenue loss
- Reduced product availability

Project FORESIGHT connects historical demand, machine learning forecasting, and inventory intelligence to help businesses understand these problems and make data-driven decisions.

---

# 🎯 Business Problem

FORESIGHT is designed to answer important retail analytics questions such as:

1. Which stores generate the highest demand?
2. Which SKUs have the strongest sales performance?
3. Which products have intermittent demand?
4. What factors influence customer demand?
5. What will demand look like over the next 30, 60 and 90 days?
6. Which store-SKU combinations face inventory risk?
7. Which locations are overstocked?
8. Where should replenishment be avoided?
9. Which products require business attention?
10. How can demand forecasting improve inventory decisions?

---

# 📊 Dataset Scale

The forecasting pipeline works with a large retail dataset containing:

| Metric | Value |
|---|---:|
| Stores | 50 |
| SKUs | 200 |
| Possible Store-SKU combinations | 10,000 |
| Historical observations | 17.65M+ |
| Historical period | 2021-01-01 to 2025-10-31 |
| Forecast horizons | 30 / 60 / 90 Days |

The dataset contains a high degree of demand intermittency, making demand classification and specialized forecasting approaches important components of the project.

> **Note:** Raw and processed datasets are intentionally excluded from the GitHub repository because of their large size.

---

# 🧠 Project Architecture

```text
Raw Retail Data
       │
       ▼
Data Loading
       │
       ▼
Data Cleaning & Preparation
       │
       ▼
Data Validation
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Demand Analysis
       │
       ▼
Demand Driver Analysis
       │
       ▼
Demand Classification
       │
       ▼
Baseline Forecasting
       │
       ▼
Intermittent Demand Forecasting
       │
       ▼
Advanced Machine Learning Forecasting
       │
       ▼
Forecast Evaluation
       │
       ▼
Final Model Selection
       │
       ▼
30 / 60 / 90 Day Future Forecast
       │
       ▼
Forecast & Inventory Integration
       │
       ▼
Inventory Risk Analysis
       │
       ▼
Inventory Recommendations
       │
       ▼
Business Insights
       │
       ▼
Interactive Streamlit Dashboard