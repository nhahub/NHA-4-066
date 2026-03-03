import pandas as pd

# ===========================
# STEP 1 - Load clean data
# ===========================
df = pd.read_csv("../data/clean_data.csv")
print(f"Loaded {len(df)} rows")

# ===========================
# STEP 2 - Keep only what we need
# We only need 3 columns:
# - response  (the answer the chatbot will give)
# - intent    (what topic it belongs to)
# - category  (the bigger group it belongs to)
# ===========================
knowledge_base = df[['response', 'intent', 'category']].copy()

# ===========================
# STEP 3 - Give every response a unique ID
# so we can track which one was retrieved
# ===========================
knowledge_base['id'] = range(len(knowledge_base))

print(f"Knowledge base has {len(knowledge_base)} entries")
print()
print("Sample:")
print(knowledge_base.head(3))

# ===========================
# STEP 4 - Save it
# ===========================
knowledge_base.to_csv("../data/knowledge_base.csv", index=False)
print()
print("Saved to data/knowledge_base.csv")