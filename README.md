# 🚀 Retail Customer Intelligence: Event-Driven MLOps Pipeline

## 📌 Project Overview
This project is an automated Machine Learning Operations (MLOps) pipeline designed for retail analytics. It transitions retail data from historical reporting to predictive (XGBoost) and prescriptive intelligence, automatically generating targeted marketing lists.

## ⚙️ Technical Architecture
* **Event-Driven Ingestion:** Uses Python `watchdog` to monitor a dropzone for new transactional data.
* **Unsupervised Learning:** `K-Means Clustering` for behavioral customer segmentation (e.g., Bargain Frequenters, Loyal Veterans).
* **Supervised Learning:** `XGBoost Classifier` to predict individual-level subscription conversion probabilities.
* **Prescriptive Engine:** Rule-based logic translating AI probabilities into precise business actions.
* **Visualization:** Deploys enriched datasets to an interactive **Microsoft Power BI** Executive Dashboard.

## 📂 Folder Structure
Create these 3 below folders in project folder.
* `/1_Raw_Dropzone` - Where daily CSV files are dropped.
* `/2_Processed_Archive` - Storage for processed raw files.
* `/3_Live_Database` - Contains the `Master_Intelligence_Data.csv` feeding Power BI.
* `auto_pipeline.py` - The core MLOps engine.

## 🚀 How to Run
1. Clone this repository.
2. Install dependencies: `pip install pandas scikit-learn xgboost watchdog`
3. Run the pipeline: `python auto_pipeline.py`
4. Drop a CSV file into the `1_Raw_Dropzone` to trigger the AI engine.
