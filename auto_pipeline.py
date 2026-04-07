import time
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import xgboost as xgb
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Defining folders
DROPZONE = "./1_Raw_Dropzone"
ARCHIVE = "./2_Processed_Archive"
DATABASE_DIR = "./3_Live_Database"
MASTER_FILE = os.path.join(DATABASE_DIR, "Master_Intelligence_Data.csv")

# Checking for folders existence
for folder in [DROPZONE, ARCHIVE, DATABASE_DIR]:
    os.makedirs(folder, exist_ok=True)

class PipelineHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Trigger only when a new CSV is dropped in the folder
        if event.is_directory or not event.src_path.endswith('.csv'):
            return
        
        filepath = event.src_path
        filename = os.path.basename(filepath)
        print(f"\n🚀 NEW DATA DETECTED: {filename}")
        print("🧎‍➡️ Initiating AI Processing Pipeline...")
        
        try:
            # Wait a second to ensure the file is fully copied before reading
            time.sleep(1) 
            self.process_data(filepath, filename)
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    def process_data(self, filepath, filename):
        # 1. READ RAW DATA
        df = pd.read_csv(filepath)
        
        # 2. FEATURE ENGINEERING
        df['Loyalty Score'] = (df['Previous Purchases'] * 2) + (df['Purchase Amount (USD)'] / 10)
        df['Estimated Lifetime Value (USD)'] = df['Purchase Amount (USD)'] * df['Previous Purchases'] * 1.5
        
        df['Value Segment'] = df.apply(lambda row: 'High Value' if row['Estimated Lifetime Value (USD)'] > 2000 else ('Medium Value' if row['Estimated Lifetime Value (USD)'] > 500 else 'Low Value'), axis=1)

        # 3. AI PERSONAS (K-MEANS)
        X_cluster = df[['Age', 'Purchase Amount (USD)', 'Previous Purchases', 'Loyalty Score']]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Customer Persona'] = pd.Series(kmeans.fit_predict(X_cluster)).map({0: 'Occasional Buyers', 1: 'Bargain Frequenters', 2: 'Loyal Veterans'})
        df['Next Best Product'] = df['Category'].apply(lambda x: "Footwear" if x == "Clothing" else ("Accessories" if x == "Footwear" else "Clothing"))

        # 4. AI PROBABILITIES (XGBOOST)
        df['Subscription_Target'] = df['Subscription Status'].apply(lambda x: 1 if x == 'Yes' else 0)
        features = ['Age', 'Purchase Amount (USD)', 'Previous Purchases', 'Review Rating', 'Loyalty Score']
        X = df[features]
        
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X, df['Subscription_Target'])
        df['Subscription_Probability'] = np.round(model.predict_proba(X)[:, 1], 2)

        # 5. PRESCRIPTIVE LOGIC
        def rules(row):
            if row['Subscription Status'] == 'No' and row['Subscription_Probability'] > 0.65:
                return f"High Priority: Target with Subscription Promo (Include {row['Next Best Product']} teaser)"
            elif row['Customer Persona'] == 'Bargain Frequenters' and row['Subscription_Probability'] > 0.20:
                return f"Send 20% Discount specifically on {row['Next Best Product']}"
            elif row['Customer Persona'] == 'Occasional Buyers' and row['Subscription_Probability'] > 0.30:
                return "Standard Nurture: Send Monthly Newsletter"
            elif row['Customer Persona'] == 'Loyal Veterans' and row['Value Segment'] == 'High Value':
                return "Upsell: Offer VIP Premium Subscription"
            return "Maintain Current Strategy"

        df['Recommended Action'] = df.apply(rules, axis=1)

        # 6. UPDATE MASTER DATABASE
        if os.path.exists(MASTER_FILE):
            # Append to existing data
            master_df = pd.read_csv(MASTER_FILE)
            updated_df = pd.concat([master_df, df], ignore_index=True)
            updated_df.to_csv(MASTER_FILE, index=False)
            print(f"✅ Added {len(df)} new records to Master Database.")
        else:
            # Create new master file
            df.to_csv(MASTER_FILE, index=False)
            print(f"✅ Created new Master Database with {len(df)} records.")

        # 7. CLEANUP: Move raw file to archive
        shutil.move(filepath, os.path.join(ARCHIVE, filename))
        print(f"📁 Archived raw file: {filename}")
        print("⏳ Pipeline standing by for new data...\n")

if __name__ == "__main__":
    print(f"📡 AI Pipeline activated. Monitoring folder: {DROPZONE}")
    event_handler = PipelineHandler()
    observer = Observer()
    observer.schedule(event_handler, DROPZONE, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1) # Keeps the script running in the background
    except KeyboardInterrupt:
        observer.stop()
    observer.join()