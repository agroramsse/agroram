import os
import re
import math

def extract_metrics_from_log(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    patterns = {
        'Length of access': r'Length of access\s+(\d+)',
        'Average returned nodes': r'average returned nodes\s+([0-9.]+)',
        'Average plaintext access': r'average plaintext access\s+([0-9.]+)',
        'Average nodes accessed': r'average of the number of nodes accessed:\s+([0-9.]+)',
        'Setup time': r'Time to create oblivious structure\s+([0-9.]+)',
        'Average round trips': r'average of the number of round trips:\s+([0-9.]+)',
        'Average search time': r'average time:\s+([0-9.]+)',
        'Average parallel time': r'average parallel time\s+([0-9.]+)',
        'Storage size': r'storage_size:\s+([0-9.]+)',
    }

    values = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        values[key] = float(match.group(1)) if match else None

    access_stats = {
        'H': int(values['Length of access']) if values['Length of access'] else 0,
        'Returned': math.ceil(values['Average returned nodes']) if values['Average returned nodes'] else 0,
        '#A': math.ceil(values['Average plaintext access']) if values['Average plaintext access'] else 0,
        '𝑂.#𝐴': math.ceil(values['Average nodes accessed']) if values['Average nodes accessed'] else 0,
    }

    timing_stats = {
        'Setup': round(values['Setup time']) if values['Setup time'] else 0,
        'Rounds': round(values['Average round trips'],1) if values['Average round trips'] else 0,
        'Search': round(values['Average search time'], 2) if values['Average search time'] else 0,
        'Par.': round(values['Average parallel time'], 3) if values['Average parallel time'] else 0,
        'Size': round(values['Storage size']) if values['Storage size'] else 0,
    }

    return access_stats, timing_stats


def format_latex_table_grouped(rows, header, caption, label):
    dataset_order = [
        "Books 1D", "gowalla 1D",
        "Spitz 2D", "cali 2D", "gowalla50 2D", "gowalla100 2D",
        "synthetic 2D-2048-d", "synthetic 2D-2048-sp", "synthetic 2D-1024",
        "nh 3D", "gowalla-3D", "synthetic 3D-128", "synthetic 3D-256"
    ]

    groups = {"1D": [], "2D": [], "3D": []}

    for dataset in dataset_order:
        if dataset not in rows:
            continue
        group_key = "1D" if "1D" in dataset else "2D" if "2D" in dataset else "3D"
        groups[group_key].append((dataset, rows[dataset]))

    latex = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{|l|{"|".join(["r"] * len(header))}|}}
\\hline
Dataset & {" & ".join(header)} \\\\
\\hline
"""

    for group in ["1D", "2D", "3D"]:
        for dataset, values in groups[group]:
            row = " & ".join(str(values[h]) for h in header)
            latex += f"{dataset} & {row} \\\\\n"
        latex += "\\hline\n"

    latex += f"""\\end{{tabular}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table}}
"""
    return latex


def format_markdown_table_grouped(rows, header):
    dataset_order = [
        "Books 1D", "gowalla 1D",
        "Spitz 2D", "cali 2D", "gowalla50 2D", "gowalla100 2D",
        "synthetic 2D-2048-d", "synthetic 2D-2048-sp", "synthetic 2D-1024",
        "nh 3D", "gowalla-3D", "synthetic 3D-128", "synthetic 3D-256"
    ] 

    groups = {"1D": [], "2D": [], "3D": []}
    for dataset in dataset_order:
        if dataset not in rows:
            continue
        group_key = "1D" if "1D" in dataset else "2D" if "2D" in dataset else "3D"
        groups[group_key].append((dataset, rows[dataset]))

    markdown = ""
    for group in ["1D", "2D", "3D"]:
        markdown += f"### {group} Datasets\n"
        markdown += "| Dataset | " + " | ".join(header) + " |\n"
        markdown += "|---------|" + "|".join(["---"] * len(header)) + "|\n"
        for dataset, values in groups[group]:
            row = " | ".join(str(values[h]) for h in header)
            markdown += f"| {dataset} | {row} |\n"
        markdown += "\n"
    return markdown


def process_all_logs_in_directory(directory="."):
    access_data = {}
    timing_data = {}

    for file in os.listdir(directory):
        if file.endswith(".log"):
            dataset = os.path.splitext(file)[0]  # Normalize names
            path = os.path.join(directory, file)
            access_stats, timing_stats = extract_metrics_from_log(path)
            access_data[dataset] = access_stats
            timing_data[dataset] = timing_stats

    access_header = ['H', 'Returned', '#A', '𝑂.#𝐴']
    timing_header = ['Setup', 'Rounds', 'Search', 'Par.', 'Size']

    print("## 📌 LaTeX Table: Access Statistics")
    print(format_latex_table_grouped(access_data, access_header,
                                     caption="Access Statistics for All Datasets",
                                     label="tab:access_all"))

    print("## 📌 LaTeX Table: Setup, Search, and Storage")
    print(format_latex_table_grouped(timing_data, timing_header,
                                     caption="Setup, Search, and Storage for All Datasets",
                                     label="tab:timing_all"))

    print("## 📊 Markdown Table: Access Statistics")
    print(format_markdown_table_grouped(access_data, access_header))

    print("## 📊 Markdown Table: Setup, Search, and Storage")
    print(format_markdown_table_grouped(timing_data, timing_header))


# Uncomment to run
process_all_logs_in_directory("log/no_early-stop/")
