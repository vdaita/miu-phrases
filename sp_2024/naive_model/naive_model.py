import pandas as pd

# Choose the dataset (set this to either 'yes' or 'no')
dataset_option = "no"  # Change to "no" to process the 'no_filtered.csv' file

# Set file paths and output files based on the dataset option
if dataset_option == "yes":
    input_file = "yes_filtered.csv"
    output_file = "naive_model_results_yes.csv"
elif dataset_option == "no":
    input_file = "no_filtered.csv"
    output_file = "naive_model_results_no.csv"
else:
    raise ValueError("Invalid dataset_option. Choose either 'yes' or 'no'.")

# Load the data
df = pd.read_csv(input_file, index_col=[0])
print(f"Processing dataset: {input_file}")
print(f"Dataset shape: {df.shape}")

# Define keywords (subset for naive model)
keywords = [
    "made in america", "american made", "buy american", "made in usa", "usa made"
]

# Function to find matched phrases
def find_matched_phrases(text, keywords):
    if pd.isna(text):  # Handle NaN values in the Text column
        return []
    text = str(text).lower()  # Ensure text is a string and lowercase
    return [keyword for keyword in keywords if keyword in text]

# Apply the function to populate the Matched Phrases column
df['Matched Phrases'] = df['Text'].apply(lambda x: find_matched_phrases(x, keywords))

# Add a binary variable indicating whether any phrases were found
df['Phrase Found'] = df['Matched Phrases'].apply(lambda x: 1 if len(x) > 0 else 0)

# Save the results to a new CSV
df.to_csv(output_file, index=False)
print(f"Saved results to '{output_file}'")
