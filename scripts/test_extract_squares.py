import pandas as pd
df = pd.read_csv("dataset_out/squares_multi/gt.csv")
print(df["label_id"].value_counts().sort_index())
