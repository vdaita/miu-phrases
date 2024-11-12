import re
import pandas as pd
import matplotlib.pyplot as plt
import math

# Define keywords
keywords = [
    "made in america", "made in u.s.", "made in us",
    "american made", "usa made", "u.s. made", "us made",
    "buy american", "buy usa", "buy america",
    "support america", "support usa", "support u.s.",
    "patriot", "choose american", "choose usa", "choose u.s.", "choose america",
    "national pride", "usa based", "america based", "american based", "us based", "u.s. based",
    "usa produced", "america produced", "american produced", "us produced", "u.s. produced",
    "usa manufactured", "america manufactured", "american manufactured", "us manufactured", "u.s. manufactured",
    "american worker", "american job", "veteran owned", "veteran founded", "founded by veteran",
    "handcrafted in america", "handcrafted in usa", "handcrafted in u.s.", "handcrafted in us",
    "crafted in america", "crafted in u.s.", "crafted in us",
    "america heritage", "america tradition", "america value",
    "icon of america", "icon of usa", "icon of u.s.",
    "america manufactur", "u.s. manufactur"
]

# Load and process data with filtering
def load_and_process_file(file_path):
    def filter_columns(df):
        pattern = r"^\d{4}-\d{2}$"  # Regex to match YYYY-MM format
        filtered_columns = ['std_name'] + [col for col in df.columns if re.match(pattern, col)]
        return df[filtered_columns]

    # Load only the first 100 rows for testing
    df = pd.read_csv(file_path, low_memory=False)
    df = filter_columns(df)
    df = df.drop_duplicates(subset='std_name', keep='first').reset_index(drop=True)
    df = df.drop(columns=['std_name'])
    
    # Convert column names to datetime for time series plotting
    df.columns = pd.to_datetime(df.columns, format='%Y-%m')
    return df

# Keyword counting functions
def count_keywords_in_cell(cell, keywords):
    return sum(1 if keyword.lower() in str(cell).lower() else 0 for keyword in keywords)

def calculate_total_counts(df, columns, keywords):
    total_counts_df = pd.DataFrame(index=df.index, columns=columns, dtype=int).fillna(0)
    for index, row in df.iterrows():
        previous_count = 0
        for col_idx, col in enumerate(reversed(columns)):
            cell_value = row[col]
            current_count = count_keywords_in_cell(cell_value, keywords)
            if current_count == 0 and previous_count > 0:
                total_counts_df.at[index, columns[len(columns) - 1 - col_idx]] = previous_count
            else:
                total_counts_df.at[index, columns[len(columns) - 1 - col_idx]] = current_count
                previous_count = current_count
    return total_counts_df

# Function to run analysis for each file
def run_keyword_analysis(file_paths):
    all_tf_idf_totals = {}
    min_length = None
    
    for file_path in file_paths:
        print("Processing file:", file_path)
        df = load_and_process_file(file_path)
        columns = list(df.columns)
        total_counts_df = calculate_total_counts(df, columns, keywords)
        
        # Sum each column to get yearly totals
        tf_idf_total = total_counts_df.sum(axis=0).tolist()
        
        # Track minimum length to ensure alignment
        if min_length is None or len(tf_idf_total) < min_length:
            min_length = len(tf_idf_total)
        
        # Adjust label based on file name
        if "about_us" in file_path:
            label = "About Us No TF-IDF"
        elif "company_website" in file_path:
            label = "Company Website Information No TF-IDF"
        else:
            label = file_path  # Fallback in case of new names

        all_tf_idf_totals[label] = tf_idf_total
        print("Completed processing file:", file_path)
    
    return all_tf_idf_totals, min_length, columns

# Plotting function for time series data
def plot_results(all_tf_idf_totals, columns, min_length):
    # Trim columns to match the minimum length
    columns_trimmed = columns[-min_length:]
    
    # Trim each tf_idf_total to min_length and prepare combined total
    trimmed_totals = {}
    combined_total = [0] * min_length
    for label, tf_idf_total in all_tf_idf_totals.items():
        trimmed = tf_idf_total[-min_length:]
        trimmed_totals[label] = trimmed
        combined_total = [sum(x) for x in zip(combined_total, trimmed)]

    # Plot each individual dataset and combined total
    plt.figure(figsize=(10, 6))
    colors = {'About Us No TF-IDF': 'orange', 'Company Website Information No TF-IDF': 'green'}
    for label, trimmed in trimmed_totals.items():
        plt.plot(columns_trimmed, trimmed, label=label, color=colors.get(label, 'black'))
    
    plt.plot(columns_trimmed, combined_total, label="Combined No TF-IDF", linestyle='-', color='blue')
    plt.legend()
    plt.title('Non-TF-IDF Values Over Time')
    plt.xlabel('Year')
    plt.ylabel('Sum')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Add the function to generate final values by year, similar to countcsvalex.py
def generate_final_value_by_year(df, total_counts_df, columns, document_count_sum, data):
    import math
    import pandas as pd  # Ensure pandas is imported in the function's scope

    def count_keyword_in_cell(cell, keyword):
        return 1 if keyword.lower() in str(cell).lower() else 0

    def count_total_keywords(cell, keywords):
        return sum(count_keyword_in_cell(cell, kw) for kw in keywords)
    keyword, term_existence = data

    print("Processing keyword: ", keyword)
    term_existence_full = [0] * len(columns)

    if term_existence == 0: # Skip over if this term wasn't counted in any of the years
        return {keyword: term_existence_full}

    for index, row in df.iterrows():
        previous_value = 0
        previous_total_count = 0
        for col_idx, col in enumerate(reversed(columns)):
            cell_value = row[col]
            current_total_count = total_counts_df.at[index, col]

            if current_total_count == 0 and previous_total_count > 0:
                # If total count drops to 0 but was higher before, carry over the previous value
                term_existence_full[len(columns) - 1 - col_idx] += previous_value
            elif pd.isna(cell_value) or isinstance(cell_value, int):
                term_existence_full[len(columns) - 1 - col_idx] += 0
                previous_value = 0
            else:
                # if keyword in cell_value.lower():
                #     term_existence[len(columns) - 1 - col_idx] += 1
                # else:
                #     term_existence[len(columns) - 1 - col_idx] += 0


                idf = math.log(1 + (document_count_sum/term_existence)) # This is the IDF formula
                keyword_count = row[col].lower().count(keyword) # This is the term frequency within this document
                value = (keyword_count * idf / len(row[col]))*100 # This is the formula

                # keyword_count = row[col].lower().count(keyword) # This is the term frequency within this document (for not using TF-IDF)
                # value = keyword_count # No changes to the value here! (for not using TF-IDF)

                term_existence_full[len(columns) - 1 - col_idx] += value
                previous_value = value

            if current_total_count == 0 and previous_total_count > 0:
                previous_total_count = previous_total_count
            else:
                previous_total_count = current_total_count  # Update the total count for the next iteration


    return {keyword: term_existence_full}

# Main execution
if __name__ == "__main__":
    # Updated file paths with consistent naming
    file_paths = [
        "about_us_second_round_with_additional_firms.csv",
        "company_website_second_round_with_additional_firms.csv"
    ]
    all_tf_idf_totals, min_length, columns = run_keyword_analysis(file_paths)
    plot_results(all_tf_idf_totals, columns, min_length)

    # Initialize variables to store year sums without TF-IDF
    all_year_sums_no_tfidf = {}
    min_length_no_tfidf = None

    for file_path in file_paths:
        print("Processing file:", file_path)
        df = load_and_process_file(file_path)
        columns = list(df.columns)
        total_counts_df = calculate_total_counts(df, columns, keywords)

        # Generate year_sums_no_tfidf for each file using the generate_final_value_by_year function
        term_count = {keyword: sum(
            total_counts_df.apply(lambda row: count_keywords_in_cell(row, [keyword]), axis=1)
        ) for keyword in keywords}

        # Use generate_final_value_by_year in non-TF-IDF mode
        data = [(keyword, term_count[keyword]) for keyword in keywords]
        tf_idf_total = {}
        for item in data:
            result = generate_final_value_by_year(df, total_counts_df, columns, document_count_sum=0, data=item)
            tf_idf_total.update(result)

        # Sum the values for all keywords to get the year sums
        year_sums_no_tfidf = [0] * len(columns)
        for keyword in keywords:
            for idx in range(len(columns)):
                year_sums_no_tfidf[idx] += tf_idf_total[keyword][idx]

        # Adjust label based on file name
        if "about_us" in file_path:
            label = "About Us No TF-IDF"
        elif "company_website" in file_path:
            label = "Company Website Information No TF-IDF"
        else:
            label = file_path  # Fallback in case of new names

        all_year_sums_no_tfidf[label] = year_sums_no_tfidf

        # Update min_length_no_tfidf
        if min_length_no_tfidf is None or len(year_sums_no_tfidf) < min_length_no_tfidf:
            min_length_no_tfidf = len(year_sums_no_tfidf)

        print("Completed processing file:", file_path)

    # Trim columns and data to match minimum length
    columns_trimmed = columns[-min_length_no_tfidf:]

    year_sums_about_us_no_tfidf_trimmed = all_year_sums_no_tfidf['About Us No TF-IDF'][-min_length_no_tfidf:]
    year_sums_company_website_information_no_tfidf_trimmed = all_year_sums_no_tfidf['Company Website Information No TF-IDF'][-min_length_no_tfidf:]

    # Combine the data
    combined_no_tfidf = [sum(x) for x in zip(year_sums_about_us_no_tfidf_trimmed, year_sums_company_website_information_no_tfidf_trimmed)]

    # Plot the trimmed versions
    plt.figure(figsize=(10, 6))
    plt.plot(columns_trimmed, year_sums_about_us_no_tfidf_trimmed, label='About Us No TF-IDF', linestyle='-', color='orange')
    plt.plot(columns_trimmed, year_sums_company_website_information_no_tfidf_trimmed, label='Company Website Information No TF-IDF', linestyle='-', color='green')
    plt.plot(columns_trimmed, combined_no_tfidf, label='Combined No TF-IDF', linestyle='-', color='blue')
    plt.legend()
    plt.title('Non-TF-IDF Values Over Time')
    plt.xlabel('Year')
    plt.ylabel('Sum')
    plt.show()
