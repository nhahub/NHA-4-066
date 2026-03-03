import pandas as pd

# ===========================
# STEP 1 - Load the raw data
# ===========================
df = pd.read_csv("../data/raw_data/customer_support_dataset.csv")
print(f"Loaded {len(df)} rows")

# ===========================
# STEP 2 - Remove empty rows
# ===========================
df = df.dropna()
print(f"After removing empty rows: {len(df)} rows")

# ===========================
# STEP 3 - Remove duplicate rows
# ===========================
df = df.drop_duplicates()
print(f"After removing duplicates: {len(df)} rows")

# ===========================
# STEP 4 - Clean the text
# (remove extra spaces, lowercase)
# ===========================
df['instruction'] = df['instruction'].str.strip().str.lower()
df['response'] = df['response'].str.strip()

print("Text cleaned successfully")

# ===========================
# STEP 5 - Save the clean data
# ===========================
df.to_csv("../data/clean_data.csv", index=False)
print("Clean data saved to data/clean_data.csv")