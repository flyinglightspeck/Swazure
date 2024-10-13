import json

from utils import get_value_by_name

beta_values = ["0.01", "0.1", "0.2", "0.3", "0.4", "0.5"]
part_to_idx = {"Mean": 0, "Max": 1, "std": 2}


def get_stats_by_name(rows, name):
    for row in rows:
        if row['Name'] == name:
            return [row['Mean'], row['Max'], row['std']]


def get_rows_by_args(data, category, tech, shape, beta):
    if tech == 'Move Source':
        return data[category][tech][shape]["2"][beta]
    else:
        return data[category][tech][shape][beta]


def export_metric(data, opt_method, technique, shapes, stat, part, get_value=False):
    result = {shape: list() for shape in shapes}

    for shape in shapes:
        for beta in beta_values:
            rows = get_rows_by_args(data, opt_method, technique, shape, beta)
            if get_value:
                value = get_value_by_name(rows, stat)
            else:
                value = get_stats_by_name(rows, stat)[part_to_idx[part]]
            result[shape].append(f"{{{beta}, {value}}}")

    return result


def print_results_in_mathematica_format(results):
    print("<|" + ",".join([f"\"{shape}\"->{{{','.join(list(sorted(data)))}}}" for shape, data in results.items()]) + "|>")


def save_as_json(path, results):
    with open(path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    src = "../data/latest/data_frames.json"
    with open(src, "r") as file:
        data = json.load(file)

    hop_results1 = export_metric(data, "hops", "Move Obstructing", ["Kangaroo"], "Total Times Solution Applied", "Mean", get_value=True)
    hop_results2 = export_metric(data, "hops", "Move Obstructing", ["Kangaroo"], "Total Times Solution Worked", "Mean", get_value=True)
    hop_results3 = export_metric(data, "hops", "Move Source (t=2)", ["Kangaroo"], "Total Times Solution Applied", "Mean", get_value=True)
    hop_results4 = export_metric(data, "hops", "Move Source (t=2)", ["Kangaroo"], "Total Times Solution Worked", "Mean", get_value=True)
    # hop_results1 = export_metric(data, "hops", "Shortest Path", ["Dragon", "Skateboard"], "Unreachable Blind Fls Pairs", "Mean", get_value=True)
    # hop_results2 = export_metric(data, "hops", "Shortest Path", ["Dragon", "Skateboard"], "Total Blind Fls Pairs", "Mean", get_value=True)
    # weight_results = export_metric(data, "length", "Shortest Path", ["Kangaroo"], "Unreachable Blind Fls Pairs", "Mean", get_value=True)
    # weight_results = export_metric(data, "length", "Shortest Path", ["Racecar"], "Phi Error (degree)", "Max", get_value=False)


    print_results_in_mathematica_format(hop_results2)
    print_results_in_mathematica_format(hop_results1)
    print_results_in_mathematica_format(hop_results4)
    print_results_in_mathematica_format(hop_results3)
    # print_results_in_mathematica_format(weight_results)
