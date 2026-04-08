import pandas as pd
import numpy as np

attack = pd.read_csv("datasets/attack_flows.csv")
normal = pd.read_csv("datasets/normal_flows.csv")

# Clean invalid values
for df in (attack, normal):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

# Remove obvious local/noisy sources if needed
# Keep as-is for now

n = min(len(normal), len(attack))

attack_bal = attack.sample(n=n, random_state=42)
normal_bal = normal.sample(n=n, random_state=42)

full = pd.concat([attack_bal, normal_bal], ignore_index=True)
full = full.sample(frac=1, random_state=42).reset_index(drop=True)

full.to_csv("datasets/final_balanced_flows.csv", index=False)

print("Attack original:", len(attack))
print("Normal original:", len(normal))
print("Balanced attack:", len(attack_bal))
print("Balanced normal:", len(normal_bal))
print("Final dataset:", len(full))
print("Saved: datasets/final_balanced_flows.csv")
