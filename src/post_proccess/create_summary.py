import csv
import json
import sys
from utils import get_rows_by_args, get_value_by_name


def get_stats_by_name(rows, name):
    for row in rows:
        if row['Name'] == name:
            return [row[stat] for stat in stat_parts]


def compare(stat1, stat_ref):
    return [
        stat1[0],
        1 - (stat1[0] / stat_ref[0]),
        stat1[1],
        1 - (stat1[1] / stat_ref[1]),
        stat1[2],
        1 - (stat1[2] / stat_ref[2]),
    ] if len(stat_parts) == 3 else [
        stat1[0],
        1 - (stat1[0] / stat_ref[0]),
        stat1[1],
        1 - (stat1[1] / stat_ref[1]),
        stat1[2],
        1 - (stat1[2] / stat_ref[2]),
        stat1[3],
        1 - (stat1[3] / stat_ref[3]),
    ]


def create_summary(src):
    with open(src, "r") as file:
        data = json.load(file)

    header = [
        "Optimized Metric",
        "Technique",
        "Shape",
        "Beta",
        # "Min Dist Err", "diff",
        "Avg Dist Err", "diff", "Max Dist Err", "diff", "Std Dist Err", "diff",
        # "Min Angle Err", "diff",
        "Avg Angle Err", "diff", "Max Angle Err", "diff", "Std Angle Err", "diff",
        # "Min Theta Err", "diff",
        "Avg Theta Err", "diff", "Max Theta Err", "diff", "Std Theta Err", "diff",
        # "Min Phi Err", "diff",
        "Avg Phi Err", "diff", "Max Phi Err", "diff", "Std Phi Err", "diff",
    ]

    summary_rows = [header]

    for opt_method, data_technique in data.items():
        for tech, data_shapes in data_technique.items():
            for shape, data_beta in data_shapes.items():
                # if tech == "Move Source (t=2)":
                #     data_beta = data_beta["2"]
                # elif tech == "Move Source (t=10)":
                #     data_beta = data_beta["10"]
                for beta, rows in data_beta.items():
                    # print(opt_method, tech, shape, beta)
                    summary_rows.append([
                        opt_method,
                        tech,
                        shape,
                        beta,
                        *compare(get_stats_by_name(rows, "Distance Error (%)"),
                                 get_stats_by_name(get_rows_by_args(data, opt_method, "Shortest Path", shape, beta),
                                                   "Distance Error (%)")),
                        *compare(get_stats_by_name(rows, "Angle Error (degree)"),
                                 get_stats_by_name(get_rows_by_args(data, opt_method, "Shortest Path", shape, beta),
                                                   "Angle Error (degree)")),
                        *compare(get_stats_by_name(rows, "Theta Error (degree)"),
                                 get_stats_by_name(get_rows_by_args(data, opt_method, "Shortest Path", shape, beta),
                                                   "Theta Error (degree)")),
                        *compare(get_stats_by_name(rows, "Phi Error (degree)"),
                                 get_stats_by_name(get_rows_by_args(data, opt_method, "Shortest Path", shape, beta),
                                                   "Phi Error (degree)")),
                    ])
    return summary_rows


def create_summary_hops(src):
    with open(src, "r") as file:
        data = json.load(file)

    header = [
        "Optimized Metric",
        "Technique",
        "Shape",
        "Beta",
        "Min Hops", "diff", "Mean Hops", "diff", "Max Hops", "diff", "std Hops", "diff",
        "Min Length", "diff", "Mean Length", "diff", "Max Length", "diff", "std Length", "diff",
    ]

    summary_rows = [header]

    for opt_method, data_technique in data.items():
        for tech, data_shapes in data_technique.items():
            for shape, data_beta in data_shapes.items():
                if tech == "Averaging":
                    continue
                # if tech == "Move Source (t=2)":
                #     data_beta = data_beta["2"]
                # elif tech == "Move Source (t=10)":
                #     data_beta = data_beta["10"]
                for beta, rows in data_beta.items():
                    # print(opt_method, tech, shape, beta)
                    summary_rows.append([
                        opt_method,
                        tech,
                        shape,
                        beta,
                        *compare(get_stats_by_name(rows, "Hops of All Paths"),
                                 get_stats_by_name(get_rows_by_args(data, opt_method, "Shortest Path", shape, beta),
                                                   "Hops of All Paths")),
                        *compare(get_stats_by_name(rows, "Weight of All Paths"),
                                 get_stats_by_name(get_rows_by_args(data, opt_method, "Shortest Path", shape, beta),
                                                   "Weight of All Paths")),
                    ])
    return summary_rows


def create_summary_sanity(src):
    with open(src, "r") as file:
        data = json.load(file)

    header = [
        "Optimized Metric",
        "Technique",
        "Shape",
        "Beta",
        "Min Shortest Distance Between Pairs (cm)",
        "Avg Shortest Distance Between Pairs (cm)",
        "Max Shortest Distance Between Pairs (cm)",
        "Computed Radius (cm)",
        "Removed Points",
        "Unreachable Fuzzy Fls Pairs",
        "Total Fuzzy Fls Pairs",
        "Kissing Neighbors",
    ]

    summary_rows = [header]

    for opt_method, data_technique in data.items():
        for tech, data_shapes in data_technique.items():
            for shape, data_beta in data_shapes.items():
                # if tech == "Averaging":
                #     continue
                # if tech == "Move Source (t=2)":
                #     data_beta = data_beta["2"]
                # elif tech == "Move Source (t=10)":
                #     data_beta = data_beta["10"]
                for beta, rows in data_beta.items():
                    # print(opt_method, tech, shape, beta)
                    summary_rows.append([
                        opt_method,
                        tech,
                        shape,
                        beta,
                        *get_stats_by_name(rows, "Shortest Distance Between Pairs (cm)")[:3],
                        get_value_by_name(rows, "Computed Radius (cm)"),
                        get_value_by_name(rows, "Removed Points"),
                        get_value_by_name(rows, "Unreachable Fuzzy FLS Pairs"),
                        get_value_by_name(rows, "Total Fuzzy FLS Pairs"),
                    ])
    return summary_rows


def create_summary_solutions(src):
    with open(src, "r") as file:
        data = json.load(file)

    header = [
        "Optimized Metric",
        "Technique",
        "Shape",
        "Beta",
        "# Fuzzy Pairs",
        "# Fuzzy Pair w/out Pose",
        "# Solution Applied",
        "# Solution Worked",
        'Min Total Dist Moved (cm)',
        'Mean Total Dist Moved (cm)',
        'Max Total Dist Moved (cm)',
        'std Total Dist Moved (cm)',
        'Min Distance Error (%)',
        'Mean Distance Error (%)',
        'Max Distance Error (%)',
        'std Distance Error (%)',
        'Min Angle Error (degree)',
        'Mean Angle Error (degree)',
        'Max Angle Error (degree)',
        'std Angle Error (degree)',
        'Min Theta Error (degree)',
        'Mean Theta Error (degree)',
        'Max Theta Error (degree)',
        'std Theta Error (degree)',
        'Min Phi Error (degree)',
        'Mean Phi Error (degree)',
        'Max Phi Error (degree)',
        'std Phi Error (degree)',
    ]

    summary_rows = [header]

    for opt_method, data_technique in data.items():
        for tech, data_shapes in data_technique.items():
            for shape, data_beta in data_shapes.items():
                if tech == "Averaging":
                    continue
                if "Move Source" in tech:
                    # data_beta = data_beta["2"]
                    dist_moved_metric = "Total Distance Source FLS Moved (cm)"
                elif tech == "Move Obstructing":
                    dist_moved_metric = "Total Distance Blocking FLSs Moved (cm)"
                else:
                    continue
                for beta, rows in data_beta.items():
                    # print(opt_method, tech, shape, beta)
                    summary_rows.append([
                        opt_method,
                        tech,
                        shape,
                        beta,
                        get_value_by_name(rows, "Total Fuzzy FLS Pairs"),
                        get_value_by_name(rows, "Unreachable Fuzzy FLS Pairs"),
                        get_value_by_name(rows, "Total Times Solution Applied"),
                        get_value_by_name(rows, "Total Times Solution Worked"),
                        *get_stats_by_name(rows, dist_moved_metric),
                        *get_stats_by_name(rows, "Distance Error (%)"),
                        *get_stats_by_name(rows, "Angle Error (degree)"),
                        *get_stats_by_name(rows, "Theta Error (degree)"),
                        *get_stats_by_name(rows, "Phi Error (degree)"),
                    ])
    return summary_rows


if __name__ == "__main__":
    # stat_parts = ["Mean", "Max", "std"]
    stat_parts = ["Min", "Mean", "Max", "std"]

    # funcs = [create_summary]
    # names = ['summary']

    funcs = [
        create_summary_solutions,
        create_summary_hops,
        create_summary_sanity
    ]
    names = [
        'summary_mo_vs_mb',
        'summary_hops',
        'summary_sanity'
    ]

    for func, name in zip(funcs, names):
        rows = func("../data/latest/data_frames.json")

        with open(f'../data/latest/{name}.csv', 'w', newline='') as file:
            writer = csv.writer(file)

            for row in rows:
                writer.writerow(row)
