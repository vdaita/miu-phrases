import matplotlib.pyplot as plt
import json
import pandas as pd
import os
from fire import Fire

def graph_list(category_names, subset_title):
    data_x = {}
    data_y = {}

    for category_name in category_names:
        with open(f"graph_data/{category_name}.json") as f:
            print(f"    Loading {category_name}")
            data = json.load(f)
            data_x[category_name] = data["columns"]
            data_y[category_name] = data["year_sums"]
            data_x[category_name] = [pd.to_datetime(year, format='mixed') for year in data_x[category_name]]
            plt.plot(data_x[category_name], data_y[category_name], label=category_name)

    plt.xlabel('Year')
    plt.ylabel('Average Keyword Count')
    plt.title(f'Average Keyword Count by Year For Subset {subset_title}')
    plt.grid(True)
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.legend()
    plt.savefig(f"keyword_charts/combined_{subset_title}.png")
    plt.clf()

def main(words_json_file: str = "generated_words.json"):
    with open(words_json_file) as f:
        categories_original = json.load(f)
    
    categories = []
    for category_type in categories_original:
        for category_name, keyword_list in categories_original[category_type].items():
            categories.append(category_name + "_" + category_type)

    graph_list(categories, "all")
    print(categories)
    graph_list([category_name for category_name in categories if category_name.endswith("_refined")], "refined")
    print([category_name for category_name in categories if category_name.endswith("_refined")])
    graph_list([category_name for category_name in categories if category_name.endswith("_unrefined")], "unrefined")
    print([category_name for category_name in categories if category_name.endswith("_unrefined")])

if __name__ == "__main__":
    Fire(main)