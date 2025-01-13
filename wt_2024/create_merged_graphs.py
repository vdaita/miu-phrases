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
    plt.ylabel('Averaged TF-IDF Score Across Documents')
    plt.title(f'Merged TF-IDF Scores for Subset {subset_title}')
    plt.grid(True)
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.legend()
    plt.savefig(f"keyword_charts/combined_{subset_title}.png")
    plt.clf()

def main(words_json_file: str = "generated_words.json"):
    with open(words_json_file) as f:
        categories_original = json.load(f)
    
    categories = []
    metric_types = ["", "_simple_ratio"]
    for metric_type in metric_types:
        for category_type in categories_original:
            for category_name, keyword_list in categories_original[category_type].items():
                categories.append(category_name + "_" + category_type + metric_type)

    all_tf_idf_list = [category_name for category_name in categories if not(category_name.endswith("_ratio"))]
    graph_list(all_tf_idf_list, "all_tf_idf")
    print(all_tf_idf_list)

    refined_tf_idf_list = [category_name for category_name in categories if category_name.endswith("_refined")]
    graph_list(refined_tf_idf_list, "refined_tf_idf")
    print(refined_tf_idf_list)

    unrefined_tf_idf_list = [category_name for category_name in categories if category_name.endswith("_unrefined")]
    graph_list(unrefined_tf_idf_list, "unrefined_tf_idf")
    print(unrefined_tf_idf_list)
    
    refined_ratio_list = [category_name for category_name in categories if category_name.endswith("_refined_simple_ratio")]
    graph_list(refined_ratio_list, "refined_simple_ratio")
    print(refined_ratio_list)

    unrefined_ratio_list = [category_name for category_name in categories if category_name.endswith("_unrefined_simple_ratio")]
    graph_list(unrefined_ratio_list, "unrefined_simple_ratio")
    print(unrefined_ratio_list)

if __name__ == "__main__":
    Fire(main)