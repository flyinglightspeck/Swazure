import ast
import concurrent.futures
import csv
import json
import os
import re

import pandas as pd
import numpy as np

summary = {
    "hops": {
        "Shortest Path": {},
        "Move Obstructing": {},
        "Move Obstructing+": {},
        "Move Source (t=2)": {},
        "Move Source+ (t=2)": {},
        "Move Source (t=10)": {},
        "Move Source+ (t=10)": {},
        "Averaging": {},
    },
    "length": {
        "Shortest Path": {},
        "Move Obstructing": {},
        "Move Obstructing+": {},
        "Move Source+ (t=2)": {},
        "Move Source (t=2)": {},
        "Move Source+ (t=10)": {},
        "Move Source (t=10)": {},
        "Averaging": {},
    },
}
beta_pattern = r'b(\d+\.\d+)'
threshold_pattern = r'_t(\d+)'
shapes = ['chess', 'palm', 'kangaroo', 'dragon', 'skateboard', 'racecar']
techniques = [
    ('single', 'Shortest Path'),
    ('move_blocking+', 'Move Obstructing+'),
    ('move_blocking', 'Move Obstructing'),
    ('move_source+', 'Move Source+'),
    ('move_source', 'Move Source'),
    ('dual', 'Averaging'),
]


def get_col_stats(name, col):
    return {
        "Name": name,
        "Value": None,
        "Min": col.min(),
        "Mean": col.mean(),
        "Max": col.max(),
        "std": col.std()
    }


def get_beta_value(filename):
    match = re.search(beta_pattern, filename)

    if match:
        floating_number = match.group(1)
        return floating_number
    else:
        raise Exception("Could not extract beta value")


def get_t_value(filename):
    match = re.search(threshold_pattern, filename)

    if match:
        floating_number = match.group(1)
        return floating_number
    else:
        raise Exception("Could not extract beta value")


def get_shape_name(filename):
    for shape in shapes:
        if shape in filename:
            return shape.capitalize()


def get_technique_name(filename):
    for given, nickname in techniques:
        if given in filename:
            if given == 'move_source' or given == 'move_source+':
                if '_t2' in filename:
                    return nickname + ' (t=2)'
                elif '_t10' in filename:
                    return nickname + ' (t=10)'
            return nickname


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)  # inclination
    phi = np.arctan2(y, x)  # azimuth
    return r, np.degrees(theta), np.degrees(phi)


def add_theta_and_phi(df):
    col_name = "Computed C2C Relative Pose (cm)"
    col_name = col_name if col_name in df.columns else "Computed C2C Avg Relative Pose (cm)"
    # Convert string representation of 3D vectors to numpy arrays
    df[col_name] = df[col_name].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=' '))
    df["Ground Truth C2C Relative Pose (cm)"] = df["Ground Truth C2C Relative Pose (cm)"].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=' '))

    # Convert Cartesian coordinates to Spherical coordinates
    df[['r_A', 'theta_A', 'phi_A']] = df[col_name].apply(
        lambda v: pd.Series(cartesian_to_spherical(*v)))
    df[['r_B', 'theta_B', 'phi_B']] = df["Ground Truth C2C Relative Pose (cm)"].apply(
        lambda v: pd.Series(cartesian_to_spherical(*v)))

    # df[['theta_A']] = df[col_name].apply(
    #     lambda v: pd.Series(cartesian_to_spherical(*v)[1:2]))
    # df[['theta_B']] = df["Ground Truth C2C Relative Pose (cm)"].apply(
    #     lambda v: pd.Series(cartesian_to_spherical(*v)[1:2]))

    # Calculate the differences between theta and phi
    df['Theta Error (degree)'] = np.abs(df['theta_A'] - df['theta_B'])
    df['Phi Error (degree)'] = np.abs(df['phi_A'] - df['phi_B'])

    # Ensure phi_diff is within [0, 180]
    df['Phi Error (degree)'] = df['Phi Error (degree)'].apply(lambda x: min(x, 360 - x))

    # Optionally, drop the intermediate spherical coordinate columns
    # df = df.drop(columns=['theta_A', 'theta_B'])
    df = df.drop(columns=['r_A', 'theta_A', 'phi_A', 'r_B', 'theta_B', 'phi_B'])
    #
    return df


def add_gt_dir(df):
    col_name = "Computed C2C Relative Pose (cm)"
    df[col_name] = df[col_name].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=' '))

    df[['gt_dir']] = df[col_name].apply(
        lambda v: pd.Series(['x', 'y', 'z'][np.abs(v).argmax()]))

    value_counts = df['gt_dir'].value_counts()
    total_values = value_counts.sum()
    value_shares = value_counts / total_values

    print(f"Total values: {total_values}")
    print(f"Values Count: {value_counts}")
    print("Share of each value:")
    print(value_shares)
    return df


def store_summary(dest_dir):
    with open(os.path.join(dest_dir, 'data_frames.json'), 'w') as f:
        json.dump(summary, f, indent=4)


def process_a_file(args):
    root, file, src_dir, dest_dir = args
    try:
        # Create the corresponding directory in the destination
        relative_path = os.path.relpath(root, src_dir)
        dest_path = os.path.join(dest_dir, relative_path)
        os.makedirs(dest_path, exist_ok=True)

        xlsx_file_path = os.path.join(root, file)
        dest_xlsx_file_path = os.path.join(dest_path, file)

        if os.path.exists(dest_xlsx_file_path):
            return

        csv_file = file.replace('.xlsx', '_paths.csv')
        csv_file_path = os.path.join(root, 'paths', csv_file)

        technique = get_technique_name(file)
        shape = get_shape_name(file)
        beta = get_beta_value(file)
        category = "length" if "_w_" in file else "hops"

        # print(category, technique, shape, beta)
        print(xlsx_file_path)
        # continue
        results = summary[category][technique]
        if shape not in results:
            results[shape] = {}
        results = results[shape]

        xlsx_df = pd.ExcelFile(xlsx_file_path)
        csv_df = pd.read_csv(csv_file_path)

        new_csv_df = add_theta_and_phi(csv_df)
        new_csv_df.to_csv(csv_file_path, index=False)

        # Add metric to stats sheet
        with pd.ExcelWriter(dest_xlsx_file_path, engine='openpyxl') as writer:
            for sheet_name in xlsx_df.sheet_names:
                df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name)
                if sheet_name == "stats":
                    new_rows = [
                        get_col_stats("Theta Error (degree)", new_csv_df["Theta Error (degree)"]),
                        get_col_stats("Phi Error (degree)", new_csv_df["Phi Error (degree)"]),
                    ]
                    df = pd.concat([df, pd.DataFrame.from_records(new_rows)], ignore_index=True)
                    # results[beta] = df.to_dict(orient='records')
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        print(e)


def compute_gt_dir(args):
    root, file, src_dir, dest_dir = args
    try:
        # Create the corresponding directory in the destination
        relative_path = os.path.relpath(root, src_dir)

        csv_file = file.replace('.xlsx', '_paths.csv')
        csv_file_path = os.path.join(root, 'paths', csv_file)

        technique = get_technique_name(file)
        shape = get_shape_name(file)
        beta = get_beta_value(file)
        category = "length" if "_w_" in file else "hops"

        # print(category, technique, shape, beta)
        print(csv_file_path)
        # continue
        results = summary[category][technique]
        if shape not in results:
            results[shape] = {}
        results = results[shape]

        csv_df = pd.read_csv(csv_file_path)

        new_csv_df = add_gt_dir(csv_df)

    except Exception as e:
        print(e)


def process_excel_files(src_dir, dest_dir):
    # Walk through the directory structure
    args_list = []
    pool_size = 6
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.xlsx') and "move_source_racecar_3720_b0.2_t10.xlsx" in file:
                args_list.append((root, file, src_dir, dest_dir))

    print(f"Running {len(args_list)} scripts in a pool of {pool_size} processes.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=pool_size) as executor:
        results = executor.map(process_a_file, args_list)
        # results = executor.map(compute_gt_dir, args_list)

    print("All scripts have completed.")


def create_summary(src_dir, dest_dir):
    # Walk through the directory structure
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            try:
                if file.endswith('.xlsx'):
                    xlsx_file_path = os.path.join(root, file)

                    technique = get_technique_name(file)
                    shape = get_shape_name(file)
                    beta = get_beta_value(file)
                    category = "length" if "_w_" in file else "hops"

                    print(xlsx_file_path)

                    results = summary[category][technique]
                    if shape not in results:
                        results[shape] = {}
                    results = results[shape]

                    # if technique == "Move Source":
                    #     t = get_t_value(file)
                    #     if t not in results:
                    #         results[t] = {}
                    #     results = results[t]

                    df = pd.read_excel(xlsx_file_path, sheet_name="stats")
                    results[beta] = df.to_dict(orient='records')

                    # if technique == "Shortest Path":
                    #     print(results[beta])
            except Exception as e:
                print(e)


def get_path_type(path):
    if len(path) == 0:
        return "No Path"
    elif "Decaying" in path:
        return "Decaying"
    else:
        return "Sweet"


def get_path_status(args):
    root, file, src_dir = args
    try:
        csv_file_path = os.path.join(root, file)

        shape = get_shape_name(file)
        beta = get_beta_value(file)
        category = "length" if "_w_" in file else "hops"
        #
        # results = summary[category][technique]
        # if shape not in results:
        #     results[shape] = {}
        # results = results[shape]

        df = pd.read_csv(csv_file_path)
        df["Source-Target"] = df["Source"].astype(str) + "-" + df["Target"].astype(str)
        cols = [
            "Source-Target",
            "Shortest Path",
            "Edge Types",
        ]
        df = df[cols]

        df["Edge Types"] = df["Edge Types"].apply(lambda x: ast.literal_eval(x))
        df["Path Type"] = df["Edge Types"].apply(lambda x: get_path_type(x))

        return f"{category}_{shape}_{beta}", df

    except Exception as e:
        print(e)


def compare_paths(ref_df, df):
    df = df.rename(
        columns={'Shortest Path': 'Shortest Path B', 'Edge Types': 'Edge Types B', 'Path Type': 'Path Type B'})

    # print(df.shape[0], ref_df.shape[0], df.shape[0] == ref_df.shape[0])
    assert df.shape[0] == ref_df.shape[0]
    merged_df = pd.merge(ref_df, df, on='Source-Target', how='inner')

    # grouped_counts = merged_df.groupby(['Path Type', 'Path Type B']).size()
    # print(grouped_counts)

    decaying_to_nopath = len(
        merged_df[(merged_df['Path Type'] == 'Decaying') & (merged_df['Path Type B'] == 'No Path')])
    decaying_to_sweet = len(merged_df[(merged_df['Path Type'] == 'Decaying') & (merged_df['Path Type B'] == 'Sweet')])
    decaying_to_decaying = len(merged_df[(merged_df['Path Type'] == 'Decaying') & (merged_df['Path Type B'] == 'Decaying')])
    nopath_to_nopath = len(merged_df[(merged_df['Path Type'] == 'No Path') & (merged_df['Path Type B'] == 'No Path')])
    nopath_to_sweet = len(merged_df[(merged_df['Path Type'] == 'No Path') & (merged_df['Path Type B'] == 'Sweet')])
    sweet_to_sweet = len(merged_df[(merged_df['Path Type'] == 'Sweet') & (merged_df['Path Type B'] == 'Sweet')])

    return nopath_to_sweet, decaying_to_sweet, decaying_to_nopath, decaying_to_decaying, sweet_to_sweet, nopath_to_nopath


def get_general_metrics(df):
    total_blind_pairs = df.shape[0]
    pairs_without_path = len(df[(df['Path Type'] == 'No Path')])
    pairs_with_sweet_path = len(df[(df['Path Type'] == 'Sweet')])
    pairs_with_decaying_path = len(df[(df['Path Type'] == 'Decaying')])
    return total_blind_pairs, pairs_without_path, pairs_with_sweet_path, pairs_with_decaying_path


def compare_hops(src_dir):
    shortest_path_args_list = []
    move_blocking_args_list = []
    move_source_args_list = []

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.csv') and 'racecar' not in file:
                if "single" in file:
                    shortest_path_args_list.append((root, file, src_dir))
                elif "move_blocking" in file:
                    move_blocking_args_list.append((root, file, src_dir))
                elif "move_source" in file:
                    move_source_args_list.append((root, file, src_dir))

    shortest_path_dfs = {}

    rows = [
        ["Optimized Metric",
         "Technique",
         "Shape",
         "Beta",
         "Total Blind Pairs", "Pairs w/ No Path", "Pairs w/ Sweet Path", "Pairs w/ Decaying Path",
         "No Path -> Sweet", "Decaying -> Sweet", "Decaying -> No Path", "Remained Decaying", "Remained Sweet", "Remained No Path"]
    ]

    for args in shortest_path_args_list:
        file, df = get_path_status(args)
        shortest_path_dfs[file] = df
        technique = get_technique_name(args[1])
        shape = get_shape_name(args[1])
        beta = get_beta_value(args[1])
        category = "length" if "_w_" in args[1] else "hops"
        general_metrics = get_general_metrics(df)
        row = [
            category,
            technique,
            shape,
            beta,
            *general_metrics,
            '', '', '', '', '',
        ]

        rows.append(row)

    for args in move_blocking_args_list + move_source_args_list:
        file, df = get_path_status(args)
        print(args[1])
        technique = get_technique_name(args[1])
        shape = get_shape_name(args[1])
        beta = get_beta_value(args[1])
        category = "length" if "_w_" in args[1] else "hops"
        general_metrics = get_general_metrics(df)
        comp = compare_paths(shortest_path_dfs[file], df)
        row = [
            category,
            technique,
            shape,
            beta,
            *general_metrics,
            *comp
        ]

        rows.append(row)

    with open('../data/latest/summary_paths_change_2.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    src_dir = "/Users/hamed/Downloads/all_data"
    # src_dir = "/Users/hamed/Documents/Holodeck/SwarmLocator/src/data/latest"
    dest_dir = "/Users/hamed/Documents/Holodeck/SwarmLocator/src/data/latest"

    # process_excel_files(src_dir, dest_dir)
    # create_summary(src_dir, dest_dir)
    # store_summary(dest_dir)
    compare_hops(src_dir)
