import json
from glob import glob


def convert_json_to_html(json_data, output_filename="output.html"):
    """
    Converts a JSON list into a readable HTML file.

    Args:
        json_data (str or list): The JSON data as a string or a Python list.
        output_filename (str): The name of the HTML file to create.
    """
    if isinstance(json_data, str):
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError:
            print("Error: Invalid JSON string provided.")
            return
    elif isinstance(json_data, list):
        data = json_data
    else:
        print("Error: Input must be a JSON string or a Python list.")
        return

    if not isinstance(data, list):
        print("Error: The provided JSON is not a list of items.")
        return

    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JSON Data View</title>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            margin: 20px;
            background-color: #f4f7f6;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }
        .item-block {
            background-color: #e8f0fe; /* Light blue background for blocks */
            border: 1px solid #cce0ff;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        .item-block h2 {
            color: #2c3e50; /* Darker blue for main titles */
            margin-top: 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #a0c4ff;
            font-size: 1.8em;
        }
        .item-block h3 {
            color: #34495e; /* Slightly lighter blue for sub-titles */
            margin-top: 15px;
            margin-bottom: 8px;
            font-size: 1.3em;
        }
        .item-block p {
            margin-bottom: 10px;
            white-space: pre-wrap; /* Preserves line breaks and spaces */
            word-wrap: break-word; /* Breaks long words */
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #eee;
        }
        .key {
            font-weight: bold;
            color: #555;
            margin-right: 5px;
        }
        .value {
            color: #666;
        }
        .nested-block {
            margin-left: 20px;
            border-left: 3px solid #d0e0f0;
            padding-left: 10px;
            margin-top: 10px;
        }
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                padding: 15px;
            }
            .item-block {
                padding: 15px;
            }
            .item-block h2 {
                font-size: 1.5em;
            }
            .item-block h3 {
                font-size: 1.1em;
            }
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <h1>JSON Data Visualization</h1>
"""

    for i, item in enumerate(data):
        html_content += f'<div class="item-block">\n'
        html_content += f'        <h2>Item {i + 1}</h2>\n'  # Main title for each item

        for key, value in item.items():
            # Organize keys into different levels
            if key in ["id", "type"]:
                html_content += f'        <h3><span class="key">{key.replace("_", " ").title()}:</span> <span class="value">{value}</span></h3>\n'
            elif key == "content":
                html_content += f'        <h3><span class="key">Content:</span></h3>\n'
                # Preserve line breaks and format long text
                html_content += f'        <p class="value">{value}</p>\n'
            elif key == "summary":
                html_content += f'        <h3><span class="key">Summary:</span></h3>\n'
                html_content += f'        <p class="value">{value}</p>\n'
            else:
                # Handle other keys, potentially nested if they are dictionaries/lists
                html_content += f'        <div class="nested-block">\n'
                html_content += f'            <h3><span class="key">{key.replace("_", " ").title()}:</span></h3>\n'
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        html_content += f'            <p><span class="key">{sub_key.replace("_", " ").title()}:</span> <span class="value">{sub_value}</span></p>\n'
                elif isinstance(value, list):
                    html_content += f'            <ul>\n'
                    for sub_item in value:
                        if isinstance(sub_item, dict):
                            html_content += f'                <li>\n'
                            for sk, sv in sub_item.items():
                                html_content += f'                    <p><span class="key">{sk.replace("_", " ").title()}:</span> <span class="value">{sv}</span></p>\n'
                            html_content += f'                </li>\n'
                        else:
                            html_content += f'                <li><span class="value">{sub_item}</span></li>\n'
                    html_content += f'            </ul>\n'
                else:
                    html_content += f'            <p><span class="value">{value}</span></p>\n'
                html_content += f'        </div>\n'
        html_content += f'</div>\n'

    html_content += """
    </div>
</body>
</html>
"""

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML file '{output_filename}' created successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert JSON data to HTML format.")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("--output-file", type=str, default="output.html", help="Name of the output HTML file.")

    args = parser.parse_args()

    files = list(glob(args.input_file))
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            json_data = f.read()
        print(file)
        output_file = file.replace(".json", ".html")
        convert_json_to_html(json_data, output_file)
