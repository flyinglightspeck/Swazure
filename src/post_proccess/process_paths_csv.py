import json
import os
import pandas as pd
from openpyxl import load_workbook

from src.const import PathMoveBlockingCol


data_hop = {}
data_weight = {}

raw_data_hop = {}
raw_data_weight = {}


def process_path_files(src_dir):
    # Walk through the directory structure
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.csv'):
                # Create the corresponding directory in the destination
                # relative_path = os.path.relpath(root, src_dir)
                # dest_path = os.path.join(dest_dir, relative_path)
                # os.makedirs(dest_path, exist_ok=True)

                src_file_path = os.path.join(root, file)
                # dest_file_path = os.path.join(dest_path, file)

                # excel_file = pd.ExcelFile(src_file_path)

                # Extract the "paths" sheet and save it as a CSV file

                path_df = pd.read_csv(src_file_path)
                num_obstructed_path = path_df[PathMoveBlockingCol.NUM_BLOCKING_FLSS.value].ne(0).sum()
                num_total_pairs = path_df[PathMoveBlockingCol.NUM_BLOCKING_FLSS.value].count()
                # exit()

                name_parts = file.split("_")
                if "_w_" in file:
                    shape = name_parts[3].capitalize()
                    beta = name_parts[5][1:]
                    results = data_weight
                    raw_results = raw_data_weight
                else:
                    shape = name_parts[2].capitalize()
                    beta = name_parts[4][1:]
                    results = data_hop
                    raw_results = raw_data_hop

                if shape in results:
                    results[shape].append(f"{{{beta}, {100 * num_obstructed_path / num_total_pairs}}}")
                else:
                    results[shape] = [f"{{{beta}, {100 * num_obstructed_path / num_total_pairs}}}"]

                list_obstructing = path_df[PathMoveBlockingCol.NUM_BLOCKING_FLSS.value].to_list()
                if shape in raw_results:
                    raw_results[shape][beta] = list_obstructing
                else:
                    raw_results[shape] = {beta: list_obstructing}

                print(f"{shape},{beta}: {num_obstructed_path} out of {num_total_pairs}")


def print_results_in_mathematica_format(results):
    print("<|" + ",".join([f"\"{shape}\"->{{{','.join(list(sorted(data)))}}}" for shape, data in results.items()]) + "|>")


def save_as_json(path, results):
    with open(path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    src_dir = "/Users/hamed/Downloads/paths"
    process_path_files(src_dir)

    raw_data_hop = {shape: {key: raw_data_hop[shape][key] for key in sorted(raw_data_hop[shape])} for shape in raw_data_hop}
    raw_data_weight = {shape: {key: raw_data_weight[shape][key] for key in sorted(raw_data_weight[shape])} for shape in raw_data_weight}

    print("hop")
    print_results_in_mathematica_format(data_hop)

    print("weight")
    print_results_in_mathematica_format(data_weight)

    save_as_json("data/obstructing_hop.json", raw_data_hop)
    save_as_json("data/obstructing_weight.json", raw_data_weight)
