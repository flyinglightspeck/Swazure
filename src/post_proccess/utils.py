def get_value_by_name(rows, name):
    for row in rows:
        if row['Name'].lower() == name.lower():
            return row['Value']

def get_stats_by_name(rows, name):
    for row in rows:
        if row['Name'] == name:
            return [row['Mean'], row['Max'], row['std']]


def get_rows_by_args(data, category, tech, shape, beta):
    if tech == 'Move Source (t=2)':
        return data[category][tech][shape]["2"][beta]
    elif tech == 'Move Source (t=10)':
        return data[category][tech][shape]["10"][beta]
    else:
        return data[category][tech][shape][beta]
