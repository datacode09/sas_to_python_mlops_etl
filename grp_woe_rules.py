import pandas as pd
import numpy as np

# Example configuration dictionary for each variable
variable_config = {
    'prc_seg_dft_cst14': {
        'conditions': [
            ('isnull', None),
            ('<', 0.01),
            ('>=<', (0.01, 0.03)),
            ('>=<', (0.03, 0.05)),
            ('>=<', (0.05, 0.06))
        ],
        'grp_values': [2, 1, 3, 4, 5],
        'woe_values': [-0.204267134, 0.5071735376, 0.0338028823, -0.466585944, 0.446067265]
    },
    'another_variable': {
        'conditions': [
            ('isnull', None),
            ('<', 0.5),
            ('>=', 0.5)
        ],
        'grp_values': [1, 2, 3],
        'woe_values': [0.0, -0.1, 0.2]
    },
    # Add more variable configurations as needed
}

def apply_conditions(df, variable, config):
    conditions = []
    for cond, value in config['conditions']:
        if cond == 'isnull':
            conditions.append(df[variable].isnull())
        elif cond == '<':
            conditions.append(df[variable] < value)
        elif cond == '>=':
            conditions.append(df[variable] >= value)
        elif cond == '>=<':
            conditions.append((df[variable] >= value[0]) & (df[variable] < value[1]))
        # Add more condition types if needed
    

# Sample data
data = {
    'prc_seg_dft_cst14': [np.nan, 0.005, 0.02, 0.04, 0.055, 0.07],
    'another_variable': [np.nan, 0.3, 0.6, 0.8, 0.1, 0.2]
}
df = pd.DataFrame(data)

    # Apply the group and WOE values for each condition
    df[f'GRP_{variable}'] = np.select(conditions, config['grp_values'], default=np.nan)
    df[f'WOE_{variable}'] = np.select(conditions, config['woe_values'], default=np.nan)


