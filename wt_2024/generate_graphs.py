# %%
import pandas as pd
from functools import partial
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import json
from fire import Fire

# df = pd.read_csv("about_us_second_round_with_additional_firms.csv", low_memory=False)

def graph_keywords(csv_filename, keywords, keyword_name):
    df = pd.read_csv(csv_filename, low_memory=False)
    print("Opened CSV")

    # %%
    df.drop(df.columns[:14], axis=1, inplace=True)
    df = df.loc[:, ~df.columns.str.contains('\.')]
    df.head()

    # %%
    df.shape

    # %%
    columns = list(df.columns)

    # %%
    def count_keywords_in_cell(cell, keywords):
        return sum(1 if keyword.lower() in str(cell).lower() else 0 for keyword in keywords)

    def calculate_total_counts(df, columns, keywords):
        total_counts_df = pd.DataFrame(index=df.index, columns=columns, dtype=int).fillna(0)
        
        for index, row in df.iterrows():
            previous_count = 0  # Initialize the previous count
            for col_idx, col in enumerate(reversed(columns)):
                cell_value = row[col]
                current_count = count_keywords_in_cell(cell_value, keywords)
                
                # If the current count is zero and the previous count is greater than zero, use the previous count
                if current_count == 0 and previous_count > 0:
                    total_counts_df.at[index, columns[len(columns) - 1 - col_idx]] = previous_count
                else:
                    total_counts_df.at[index, columns[len(columns) - 1 - col_idx]] = current_count
                    previous_count = current_count  # Update previous count
        
        return total_counts_df

    # %%
    total_counts_df = calculate_total_counts(df, columns, keywords)

    # %%
    total_counts_df

    # %%
    document_counts = [0] * len(columns) # Initialize a list to hold the count of documents for each year.

    for row in df.itertuples(index=False):# Iterate over each row in the DataFrame.
        previous_count = 0 # Initialize the previous count to 0 for the first iteration.

        for idx in reversed(range(len(columns))):    # Iterate over the columns in reverse order to update the document count.
            value = row[idx] # Access the value using the appropriate index for itertuples() output.
            
            # Check if the current cell has a document (non-NaN and not an empty string).
            if pd.isna(value) or isinstance(value, int):
                document_counts[idx] += previous_count # If there's a document, increment the count for the year and set the previous count to 1.
            else:
                document_counts[idx] += 1 # If there's no document, add the previous year's count to this year's count.
                previous_count = 1 # We have found a document so previous should never be 0



    total_documents = sum(document_counts)
    document_count_sum = total_documents
    print(total_documents)

    term_count = {}

    def count_term_existence(keyword):
        print(f"Processing keyword: {keyword}")
        term_existence = [0] * len(columns)
        
        for index, row in df.iterrows():
            previous_total_count = 0  # Initialize the total count for all keywords in the previous cell
            for col_idx, col in enumerate(reversed(columns)):
                cell_value = row[col]
                current_total_count = total_counts_df.at[index, col]

                if current_total_count == 0 and previous_total_count > 0:
                    # If total count drops to 0 but was higher before, carry over the previous value
                    term_existence[len(columns) - 1 - col_idx] += 1
                elif pd.isna(cell_value) or isinstance(cell_value, int):
                    term_existence[len(columns) - 1 - col_idx] += 0
                else:
                    if keyword in cell_value.lower():
                        term_existence[len(columns) - 1 - col_idx] += 1
                    else:
                        term_existence[len(columns) - 1 - col_idx] += 0

                if current_total_count == 0 and previous_total_count > 0:
                    previous_total_count = previous_total_count
                else:
                    previous_total_count = current_total_count  # Update the total count for the next iteration

        return {keyword: sum(term_existence)}


    # %%
    results = [count_term_existence(keyword) for keyword in keywords]

    term_count = {}
    for result in results:
        term_count.update(result)

    print(term_count)

    # %%
    import math

    def generate_final_value_by_year(data):
        keyword, term_existence = data

        print("Processing keyword: ", keyword)
        adjusted_keyword_count = [0] * len(columns)

        if term_existence == 0: # Skip over if this term wasn't counted in any of the years
            return {keyword: adjusted_keyword_count}
        
        for row in df.itertuples(index=True):
            previous_value = 0
            for column in range(len(columns) - 1, -1, -1): # Iterate from least recent to most recent
                if pd.isna(row[column]) or type(row[column]) == int: # Does the current value here not exist?
                    adjusted_keyword_count[column] += previous_value # Add in the previous value instead
                else:

                    keyword_count = row[column].lower().count(keyword) # This is the term frequency within this document (for not using TF-IDF)
                    value = keyword_count # No changes to the value here! (for not using TF-IDF)

                    if keyword_count > 0 or value > previous_value: # Is the value greater? Does the keyword count exist
                        adjusted_keyword_count[column] += value # Increment by the value we just got
                        previous_value = value # Set previous value
                    else: 
                        adjusted_keyword_count[column] += previous_value # Just use the previous value

        return {keyword: adjusted_keyword_count}

    # %%
    results = [generate_final_value_by_year((keyword, term_count[keyword])) for keyword in keywords]

    tf_idf_total = {}
    for result in results:
        tf_idf_total.update(result)

    print(tf_idf_total)

    # %%
    print(tf_idf_total[keywords[0]])

    # %%
    year_sums = [0] * len(columns)
    for keyword in keywords: # Going through all of the keywords
        if type(tf_idf_total[keyword]) == int: # This is a redundant invalid check
            continue
        for column in range(len(columns)): # Go through all of the years
            year_sums[column] += tf_idf_total[keyword][column] # Adding up all the sums per year 
        
    for column in range(len(columns)): 
        try:
            year_sums[column] /= document_counts[column]
        except ZeroDivisionError:
            year_sums[column] /= 1
    print(year_sums[:10])

    # %%
    print(df.columns.tolist())

    # %%
    columns = pd.to_datetime(list(df.columns)) # Going back to the original dataframe and getting the columns from there
    columns = list(columns) # Turning it from Pandas format to list format
    json_columns = []
    for column in columns:
        json_columns.append(column.isoformat())

    print(len(columns), columns[:3])

    save_data = {
        "columns": json_columns,
        "year_sums": list(year_sums)
    }
    with open(f'./graph_data/{keyword_name}.json', 'w') as f:
        json.dump(save_data, f)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(columns, list(year_sums)) # Plotting it!
    plt.xlabel('Year')
    plt.ylabel('Average Keyword Count')
    plt.title(f'Average Keyword Count by Year: {keyword_name}')
    plt.grid(True)
    plt.savefig(f'.keyword_charts/{keyword_name}.png')
    # plt.show()

def main(csv_filename: str, keywords_file: str):
    with open(keywords_file) as f:
        keywords_original = json.load(f)
    keywords = {}
    for keyword_type in keywords_original:
        for keyword_name, keyword_list in keyword_type.items():
            keywords[keyword_name + "_" + keyword_type] = keyword_list
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(partial(graph_keywords), [(csv_filename, keyword_list, keyword_name) for keyword_name, keyword_list in keywords.items()]), total=len(keywords)))

if __name__ == "__main__":
    Fire(main)