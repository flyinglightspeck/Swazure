import csv
import os

import numpy as np
import pandas as pd

from add_metric import get_beta_value, get_shape_name, get_technique_name

# techniques = ['shortest', 'averaging', 'move_source', 'move_blocking']
techniques = ['shortest', 'move_blocking']
tech_to_name = {
    'shortest': 'single',
    'averaging': 'dual',
    'move_source': 'move_source',
    'move_blocking': 'move_blocking'
}
shapes = ["chess_408", "dragon_1147", "kangaroo_972", "palm_725", "skateboard_1372", "racecar_3720"]
betas = ['0.01', '0.1', '0.2', '0.3', '0.4', '0.5']

cols = ['Source', 'Target', 'Ground Truth C2C Relative Pose (cm)', 'Computed C2C Relative Pose (cm)', 'Angle Between Computed Ground Truth (degree)', 'Phi Error (degree)']

pd.set_option('display.max_columns', None)


def print_df(df):
    print("\n".join([f"{col}:\t{df[col].values[0]}" for col in cols]))


def get_cols_as_list(df):
    return [df[col].values[0] for col in cols]


def report_diff(row1, row2, metric):
    val1 = row1[metric].values[0]
    val2 = row2[metric].values[0]

    if val1 < val2:
        return "smaller"
    elif val1 > val2:
        return "larger"
    else:
        return "equal"


def get_dominant_direction(row):
    vec = row["Ground Truth C2C Relative Pose (cm)"].values[0]
    vec = np.fromstring(vec.strip('[]'), sep=' ')
    return ['x', 'y', 'z'][np.abs(vec).argmax()]


header = [
    "Technique A",
    "Technique B",
    "Shape",
    "Beta",
    *cols,
    *cols,
    *cols,
    *cols,
    "Are Pairs the Same",
    "Are Paths the Same",
    "Max Phi A vs B",
    "Max Alpha A vs B",
    "Phi of Pair with Max Phi in A vs the Same Pair in B",
    "Alpha of Pair with Max Phi in A vs the Same Pair in B",
    "Phi of Pair with Max Phi in B vs the Same Pair in A",
    "Alpha of Pair with Max Phi in B vs the Same Pair in A",
    "Direction of Pair with Max Phi in A",
    "Direction of Pair with Max Phi in B",
]

rows = [header]


def compare_max_phi(a_df_path, b_df_path, filename_a, filename_b):
    a_df = pd.read_csv(a_df_path)
    b_df = pd.read_csv(b_df_path)

    a_max_idx = a_df["Phi Error (degree)"].idxmax()
    a_max_row = a_df.loc[[a_max_idx]]
    # a_max_row
    print(f"---> pair with max phi error in {filename_a}")
    print_df(a_max_row)
    a_pair_in_b = b_df[(b_df["Source"] == a_max_row["Source"].values[0]) & (b_df["Target"] == a_max_row["Target"].values[0])]
    print(f"---> the same pair in {filename_b}")
    print_df(a_pair_in_b)
    print()

    b_max_idx = b_df["Phi Error (degree)"].idxmax()
    b_max_row = b_df.loc[[b_max_idx]]
    # b_max_row
    print(f"---> pair with max phi error in {filename_b}")
    print_df(b_max_row)
    b_pair_in_a = a_df[(a_df["Source"] == b_max_row["Source"].values[0]) & (a_df["Target"] == b_max_row["Target"].values[0])]
    print(f"---> the same pair in {filename_a}")
    print_df(b_pair_in_a)
    print()

    pair_diff = "same" if a_max_row["Source"].values[0] == b_max_row["Source"].values[0] and a_max_row["Target"].values[0] == b_max_row["Target"].values[0] else "different"
    path_diff = "same" if a_max_row["Shortest Path"].values[0] == b_max_row["Shortest Path"].values[0] else "different"

    phi_max_pair_a_compared_to_max_pair_b = report_diff(a_max_row, b_max_row, "Phi Error (degree)")
    alpha_max_pair_a_compared_to_max_pair_b = report_diff(a_max_row, b_max_row, "Angle Between Computed Ground Truth (degree)")
    phi_max_pair_a_compared_to_same_pair_b = report_diff(a_max_row, a_pair_in_b, "Phi Error (degree)")
    alpha_max_pair_a_compared_to_same_pair_b = report_diff(a_max_row, a_pair_in_b, "Angle Between Computed Ground Truth (degree)")
    phi_max_pair_b_compared_to_same_pair_a = report_diff(b_max_row, b_pair_in_a, "Phi Error (degree)")
    alpha_max_pair_b_compared_to_same_pair_a = report_diff(b_max_row, b_pair_in_a, "Angle Between Computed Ground Truth (degree)")

    max_pair_a_gt_dir = get_dominant_direction(a_max_row)
    max_pair_b_gt_dir = get_dominant_direction(b_max_row)

    rows.append([
        get_technique_name(filename_a),
        get_technique_name(filename_b),
        get_shape_name(filename_a),
        get_beta_value(filename_a),
        *get_cols_as_list(a_max_row),
        *get_cols_as_list(a_pair_in_b),
        *get_cols_as_list(b_max_row),
        *get_cols_as_list(b_pair_in_a),
        pair_diff,
        path_diff,
        phi_max_pair_a_compared_to_max_pair_b,
        alpha_max_pair_a_compared_to_max_pair_b,
        phi_max_pair_a_compared_to_same_pair_b,
        alpha_max_pair_a_compared_to_same_pair_b,
        phi_max_pair_b_compared_to_same_pair_a,
        alpha_max_pair_b_compared_to_same_pair_a,
        max_pair_a_gt_dir,
        max_pair_b_gt_dir,
    ])

    """
    three metrics:
    identity of the pairs (2)
    identity of the paths (2)
    change in phi error (3)
    change in dot angle error (3)
    dominant direction of gt pose (3)
    """


def get_file_name(tech, shape, beta):
    if tech == 'move_source':
        return f'{tech_to_name[tech]}_{shape}_b{beta}_t2_paths.csv'
    else:
        return f'{tech_to_name[tech]}_{shape}_b{beta}_paths.csv'


if __name__ == '__main__':
    src = "/Users/hamed/Downloads/all_data"

    for tech in techniques[1:]:
        for shape in shapes:
            for beta in betas:
                filename = get_file_name(tech, shape, beta)
                paths_a = os.path.join(src, tech, 'paths', filename)
                for tech_b in ["shortest"]:
                    filename_b = get_file_name(tech_b, shape, beta)
                    paths_b = os.path.join(src, tech_b, 'paths', filename_b)
                    print(f'>>>>>> {filename} vs {filename_b}:')
                    compare_max_phi(paths_a, paths_b, filename, filename_b)
                    print()
                    # exit()

    with open('../data/new/tech_comparison.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        for row in rows:
            writer.writerow(row)
