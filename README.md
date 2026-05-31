# Perishable Inventory Recommendation System

A Python project that forecasts demand for perishable products and generates inventory recommendations for retail operations.

## Project Objective

The goal is to improve perishable inventory decisions using sales, transaction, and shelf-life data. The project helps identify restocking needs, overstock risk, markdown opportunities, and demand patterns across stores and product families.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Workflow

1. Clean and transform raw sales, store, and transaction data.
2. Define perishable product families and shelf-life rules.
3. Simulate inventory batches and expiry windows.
4. Train a Random Forest model to forecast demand.
5. Generate actions such as `URGENT_RESTOCK`, `PRIORITY_RESTOCK`, `CONSIDER_MARKDOWN`, and `OK`.
6. Create charts and a report for analysis.

## Files

- `etl.py` — data cleaning and table creation
- `perishable.py` — perishable setup and inventory simulation
- `model.py` — demand forecasting
- `recommend.py` — recommendation logic
- `visualize.py` — charts and report generation
- `config.py` — file paths

## How to Run

```bash
python src/etl.py
python src/perishable.py
python src/model.py
python src/recommend.py
python src/visualize.py
```

## Why this project matters

This project shows how data analytics can support retail decision-making by reducing stockouts, highlighting overstock risk, and improving perishable inventory planning.