import pandas as pd
from functools import partial
import multiprocessing as mp

# Load CSV
df = pd.read_csv("company_website_second_round_with_additional_firms.csv", low_memory=False)
print("Opened CSV")

# Define keywords
keywords = [
    "made in america", "made in u.s.", "made in us", # made in usa and made in us overlap
    "american made", "usa made", "u.s. made", "us made",
    "buy american", "buy usa", "buy america",
    "support america", "support usa", "support u.s.",
    "patriot",
    "choose american", "choose usa", "choose u.s.", "choose america",
    "national pride",
    "usa based", "america based", "american based", "us based", "u.s. based",
    "usa produced", "america produced", "american produced", "us produced", "u.s. produced",
    "usa manufactured", "america manufactured", "american manufactured", "us manufactured", "u.s. manufactured",
    "american worker", "american job",
    "veteran owned", "veteran founded", "founded by veteran",
    "handcrafted in america", "handcrafted in usa", "handcrafted in u.s.", "handcrafted in us",
    "crafted in america", "crafted in u.s.", "crafted in us",
    "america heritage", "america tradition", "america value",
    "icon of america", "icon of usa", "icon of u.s.",
    "america manufactur", "u.s. manufactur"
]

# Prepare DataFrame and columns
df.drop(df.columns[:14], axis=1, inplace=True)
df = df.loc[:, ~df.columns.str.contains(r'\.')]
df.head()
columns = list(df.columns)

# Find the number of rows where all columns are NaN
num_rows_all_nan = df.isnull().all(axis=1).sum()
print(f"Number of rows with all NaN values: {num_rows_all_nan}")

print(f"Number of rows: {len(df)}")


# Initialize the dictionary to store example texts and matched phrases
example_dict = {}

def count_and_record_keywords_in_cell(cell, keywords):
    # Initialize list to store matched keywords
    matched_keywords = [keyword for keyword in keywords if keyword.lower() in str(cell).lower()]
    
    example_dict[str(cell)] = matched_keywords
    
    # Return the count of matched keywords
    return len(matched_keywords)

def calculate_total_counts(df, columns, keywords):
    total_counts_df = pd.DataFrame(index=df.index, columns=columns, dtype=int).fillna(0)
    
    for index, row in df.iterrows():
        previous_count = 0
        for col_idx, col in enumerate(reversed(columns)):
            cell_value = row[col]
            current_count = count_and_record_keywords_in_cell(cell_value, keywords)
            
            # If current count is zero and previous count is greater than zero, use the previous count
            if current_count < previous_count and previous_count > 0:
                total_counts_df.at[index, columns[len(columns) - 1 - col_idx]] = previous_count
            else:
                total_counts_df.at[index, columns[len(columns) - 1 - col_idx]] = current_count
                previous_count = current_count  # Update previous count
    
    return total_counts_df

# Run the function to calculate counts
total_counts_df = calculate_total_counts(df, columns, keywords)

# Convert example_dict to DataFrame and save it as a CSV
example_df = pd.DataFrame(list(example_dict.items()), columns=["Example Text", "Matched Phrases"])
example_df.to_csv("example_texts_and_matched_phrases.csv", index=False)
print("Saved example texts and matched phrases to CSV")

