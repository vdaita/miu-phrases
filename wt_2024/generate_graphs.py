# %%
import pandas as pd
from functools import partial
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import json
from fire import Fire
import numpy as np
import swifter

# df = pd.read_csv("about_us_second_round_with_additional_firms.csv", low_memory=False)

def propagate_int_value_forward(df):
    df = df.copy()
    df.mask(df <= 0, np.nan, inplace=True) # mask the values less than zero
    df = df.T.ffill(axis=0).T # forward fill so that the values that are np.nan get the closest preceding value in terms of columns - axis=1 leads to NotImplementedError
    df.fillna(0, inplace=True) # fill the remaining np.nan with 0
    return df

def graph_keywords(df, doc_exists, doc_counts, doc_counts_2d, total_words, total_words_sum_2d, keywords, keyword_name, use_tf_idf_metric: bool = True):
    final_score_sum = pd.DataFrame(0, index=df.index, columns=df.columns)

    stringified_df = df.astype(str)

    for keyword in tqdm(keywords, position=mp.current_process()._identity[0] if mp.current_process()._identity else 0):
        keyword_term_count = stringified_df.apply(lambda x: x.str.count(keyword)).fillna(0)
        keyword_term_exists = stringified_df.apply(lambda x: x.str.contains(keyword)).astype(int)
        
        keyword_term_count = propagate_int_value_forward(keyword_term_count)
        keyword_term_exists = propagate_int_value_forward(keyword_term_exists)

        keyword_term_exists = keyword_term_exists.sum(axis=0)
        keyword_term_exists_2d = pd.DataFrame(
            [keyword_term_exists.values for _ in range(len(df))],
            index=df.index,
            columns=df.columns
        )

        keyword_score = pd.DataFrame(0, index=df.index, columns=df.columns)
        mask = (keyword_term_count > 0) & (keyword_term_exists_2d > 0) & (total_words > 0)
    
        if use_tf_idf_metric:
            keyword_score[mask] = (100 * keyword_term_count[mask] * np.log(1 + (doc_counts_2d[mask] / keyword_term_exists_2d[mask])) / total_words[mask])
        else:
            keyword_score[mask] = (keyword_term_count[mask] / total_words_sum_2d[mask])
        # keyword_score[mask] = (keyword_term_count[mask] / total_words[mask])
        # keyword_score[mask] = np.log(1 + (doc_counts_2d[mask] / keyword_term_exists_2d[mask]))
        # print("Keyword score:\n", keyword_score[mask])
        # print("Document counts:\n", doc_counts_2d[mask], "\nKeyword term exists:\n", keyword_term_exists_2d[mask])

        final_score_sum += keyword_score
    # print("Finished calculating keyword term count:\n", keyword_term_count.sum(axis=0))

    tf_idf_total = final_score_sum.sum(axis=0)
    # print("Finished calculating the sum of the TF-IDF scores for each year:\n", tf_idf_total)
    if use_tf_idf_metric:
        tf_idf_total = tf_idf_total / doc_counts
    # otherwise, we are already dividing by the total number of documents there 
    # print("Finished dividing by document counts:\n", tf_idf_total)

    year_sums = tf_idf_total.tolist()

    # %%
    columns = pd.to_datetime(list(df.columns)) # Going back to the original dataframe and getting the columns from there
    columns = list(columns) # Turning it from Pandas format to list format
    json_columns = []
    for column in columns:
        json_columns.append(column.isoformat())

    save_data = {
        "columns": json_columns,
        "year_sums": list(year_sums)
    }

    if not(use_tf_idf_metric):
        keyword_name = f"{keyword_name}_simple_ratio"

    with open(f'graph_data/{keyword_name}.json', 'w') as f:
        json.dump(save_data, f)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(columns, list(year_sums)) # Plotting it!
    plt.xlabel('Year')
    plt.ylabel('Averaged TF-IDF Score Across Documents')
    plt.title(f'Average Keyword Count by Year: {keyword_name}')
    plt.grid(True)
    plt.savefig(f'keyword_charts/{keyword_name}.png')
    # plt.show()

def main(csv_filename: str = "company_website_second_round_with_additional_firms_without_redundant_cleaned.csv", keywords_file: str = "generated_words.json", use_tf_idf_metric: bool = True):
    with open(keywords_file) as f:
        keywords_original = json.load(f)
    keywords = {}
    for keyword_type in keywords_original:
        for keyword_name, keyword_list in keywords_original[keyword_type].items():
            keywords[keyword_name + "_" + keyword_type] = keyword_list
    
    df = pd.read_csv(csv_filename, low_memory=False)
    print("Opened CSV")

    # %%
    df.drop(df.columns[:14], axis=1, inplace=True)
    df = df.loc[:, ~df.columns.str.contains('\.')]
    df = df[df.columns[::-1]] # the column has the later dates earlier, so we reverse it

    # print(df.head())
    doc_exists = df.swifter.applymap(lambda x: 1 if (isinstance(x, str) and len(x) > 1 and pd.notna(x)) else 0)
    # print("Doc exists check from the dataframe: ", df.iloc[:, 0], " NA sum: ", df.iloc[:, 0].isna().sum())
    # print("Doc exists: ", doc_exists.iloc[:, 0], "Sum: ", doc_exists.iloc[:, 0].sum())
    # print("Original doc exists df dtypes: ", doc_exists.dtypes.unique())
    doc_exists = propagate_int_value_forward(doc_exists)
    # print("Doc exists: ", doc_exists.iloc[:, 0], "Sum: ", doc_exists.iloc[:, 0].sum())

    # print("Finished checking for document existence")
    doc_counts = doc_exists.sum(axis=0)
    # print("Finished calculating the sums for each column")

    # print(doc_counts)

    doc_counts_2d = pd.DataFrame(
        [doc_counts.values for _ in range(len(df))],
        index=df.index,
        columns=df.columns
    )

    total_words = df.swifter.applymap(lambda x: len(str(x).split()) if isinstance(x, str) else 0)
    total_words = propagate_int_value_forward(total_words)

    total_words_sum = total_words.sum(axis=0)
    total_words_sum_2d = pd.DataFrame(
        [total_words_sum.values for _ in range(len(df))],
        index=df.index,
        columns=df.columns
    )

    try:
        with mp.Pool(mp.cpu_count()) as pool:
            for keyword_name, keyword_list in keywords.items():
                pool.apply_async(graph_keywords, args=(df, doc_exists, doc_counts, doc_counts_2d, total_words, total_words_sum_2d, keyword_list, keyword_name), kwds={"use_tf_idf_metric": use_tf_idf_metric})
            pool.close()
            pool.join()
    except Exception as e:
        print(f"An error occurred: {e}")

    # with mp.Pool(mp.cpu_count()) as pool:
        # pool.starmap(graph_keywords, [(csv_filename, keyword_list, keyword_name) for keyword_name, keyword_list in keywords.items()])

if __name__ == "__main__":
    Fire(main)