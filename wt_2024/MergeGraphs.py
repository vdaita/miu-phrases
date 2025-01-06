import matplotlib.pyplot as plt
import json
import pandas as pd
import os

data_x = {}
data_y = {}

for file in os.listdir("graph_data"):
    with open(f"graph_data/{file}") as f:
        data = json.load(f)
        category_name = file.split(".")[0]
        data_x[category_name] = data["columns"]
        data_y[category_name] = data["year_sums"]
        data_x[category_name] = [pd.to_datetime(year, format='mixed') for year in data_x[category_name]]
        plt.plot(data_x[category_name], data_y[category_name], label=category_name)

plt.xlabel('Year')
plt.ylabel('Average Keyword Count')
plt.title('Average Keyword Count by Year')
plt.grid(True)
plt.legend()
plt.savefig("keyword_charts/combined.png")