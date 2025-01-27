import fire
import json

def extract_json_from_data_txt(input_file, output_file):
    with open(input_file, 'r') as file:
        data = file.read()

    # Split the data where it says # Generated n-grams
    sections = data.split('# Generated n-grams')

    # Create the final dictionary
    result = {
        "refined": {},
        "unrefined": {}
    }

    # Only process the section after '# Generated n-grams'
    section = sections[-1]
    current_list = []
    current_key = ""
    current_type = ""

    for line in section.splitlines():
        line = line.strip()
        if line == "":
            continue
        if line.startswith('##'):
            if current_type and current_list:
                result[current_type][current_key] = current_list
                current_list = []
            current_key = line.split(" ")[-1]
            current_type = "refined" if line.startswith('## Refined') else "unrefined"
            print(line)
        else:
            current_list.append(line)

    # Save the result to the output file
    with open(output_file, 'w') as file:
        json.dump(result, file, indent=4)

if __name__ == '__main__':
    fire.Fire(extract_json_from_data_txt)