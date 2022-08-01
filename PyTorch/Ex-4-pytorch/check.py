import pandas as pd
index = 0
data = pd.read_csv("data.csv")

index += data.first_valid_index()
path = data['filename'][index]
print(path)