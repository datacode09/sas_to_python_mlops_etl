import pandas as pd
import pandasql as ps
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Data Loading Functions

def load_fn_entity():
    try:
        logging.info("Loading fn_entity data.")
        # Replace with actual logic to load the fn_entity DataFrame
        return pd.DataFrame()  # Example placeholder
    except Exception as e:
        logging.error("Failed to load fn_entity data: %s", e)
        raise

def load_hvr_input():
    try:
        logging.info("Loading hvr_input data.")
        return pd.DataFrame()  # Example placeholder
    except Exception as e:
        logging.error("Failed to load hvr_input data: %s", e)
        raise


def load_hvr_input(file_path):
    try:
        # Log the start of the process
        logger.info("Starting to load HVR input from SAS file.")

        # Read the SAS file into a pandas DataFrame
        hvr_input = pd.read_sas(file_path, format='sas7bdat')
        logger.info("Successfully read SAS file: %s", file_path)

        # Create a copy of the input DataFrame to avoid modifying the original data
        hvr_arm = hvr_input.copy()

        # Initialize the columns for grouped and WOE values
        hvr_arm['GRP_cg6nUR'] = np.nan
        hvr_arm['WOE_cg6nUR'] = np.nan
        logger.info("Initialized GRP_cg6nUR and WOE_cg6nUR columns.")

        # Apply the grouping and WOE assignment logic
        hvr_arm['GRP_cg6nUR'] = np.where(hvr_arm['cg6nUR'].isna(), 5, hvr_arm['GRP_cg6nUR'])
        hvr_arm['WOE_cg6nUR'] = np.where(hvr_arm['cg6nUR'].isna(), -2.264105048, hvr_arm['WOE_cg6nUR'])

        hvr_arm.loc[hvr_arm['cg6nUR'] <= -0.81, ['GRP_cg6nUR', 'WOE_cg6nUR']] = [1, 0.6333727575]
        hvr_arm.loc[(hvr_arm['cg6nUR'] > -0.81) & (hvr_arm['cg6nUR'] <= -0.29), ['GRP_cg6nUR', 'WOE_cg6nUR']] = [2, 0.4320581244]
        hvr_arm.loc[(hvr_arm['cg6nUR'] > -0.29) & (hvr_arm['cg6nUR'] <= -0.21), ['GRP_cg6nUR', 'WOE_cg6nUR']] = [3, 0.311526073]
        hvr_arm.loc[hvr_arm['cg6nUR'] > -0.21, ['GRP_cg6nUR', 'WOE_cg6nUR']] = [4, -0.123880095]

        # Log the completion of the grouping and WOE logic
        logger.info("Grouping and WOE assignment logic applied successfully.")

        # Return the modified DataFrame
        return hvr_arm

    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except pd.errors.EmptyDataError:
        logger.error("The SAS file is empty: %s", file_path)
        raise
    except Exception as e:
        logger.error("An error occurred while loading or processing the file: %s", str(e))
        raise

def load_mstr_scl():
    """
    Loads the MSTR_SCL data as a DataFrame, replicating the SAS datalines block.
    """
    logging.info("Loading MSTR_SCL data.")
    try:
        data = {
            "MSTR_SCL_RTG_CD": ["I-1", "I-2", "I-3", "I-4", "I-5", "I-6", "I-7", "I-8",
                                "S-3", "S-4", "S-1", "S-2", "P-1", "P-2", "P-3", "T-1",
                                "D-1", "D-2", "D-3", "D-4"],
            "grade": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        }
        MSTR_SCL = pd.DataFrame(data)
        logging.info("MSTR_SCL data loaded successfully.")
        return MSTR_SCL
    except Exception as e:
        logging.error("Failed to load MSTR_SCL data: %s", e)
        raise

def woe_code_hvr(df):
    try:
        logging.info("Applying WOE code transformations.")
        # Placeholder for WOE transformation logic
        return df
    except Exception as e:
        logging.error("Failed to apply WOE code transformations: %s", e)
        raise


def load_cust_gen():
    try:
        logging.info("Loading cust_gen data.")
        # Replace with actual logic to load the fn_entity DataFrame
        return pd.DataFrame()  # Example placeholder
    except Exception as e:
        logging.error("Failed to load fn_entity data: %s", e)
        raise





# 2. Configuration Function to Gather Inputs

from datetime import datetime, timedelta


# Calculate the start and end dates similarly to the SAS code
def calculate_str_end_dates():
    today = datetime.today()
    
    # Calculate the start date (3 months before the current month, first day of that month)
    str_date = (today.replace(day=1) - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
    
    # Calculate the end date (1 month before the current month, last day of that month)
    last_day_of_prev_month = (today.replace(day=1) - timedelta(days=1))
    end_date = last_day_of_prev_month.strftime('%Y-%m-%d')
    
    return str_date, end_date

# Updated config function with dynamically calculated `str` and `end` dates
def config():
    """
    Configures and loads all necessary inputs for processing, with dynamically calculated `str` and `end` dates.
    """
    # Calculate dynamic start and end dates
    str_date, end_date = calculate_str_end_dates()
    
    config_data = {
        "fn_entity": load_fn_entity(),
        "hvr_input": load_hvr_input(),
        "MSTR_SCL": load_mstr_scl(),
        "cust_gen_scoreout": load_module_scoreout("cust_gen"),
         "cust_gen_input": load_cust_gen(),
             "rel": ["prc_sct_dft_cst", "prc_seg_dft_cst", "prc_reg_dft_cst", "prc_sct_bd_cst", "prc_seg_bd_cst", "prc_reg_bd_cst"],
             "abs": ["bad_customer", "default_customer", "n_sct_dft_cst", "n_seg_dft_cst", "n_reg_dft_cst", "n_sct_bd_cst", "n_seg_bd_cst", "n_reg_bd_cst"],
             "coal": ["bad_customer", "default_customer", "prc_sct_dft_cst", "prc_seg_dft_cst", "prc_reg_dft_cst", "prc_sct_bd_cst", "prc_seg_bd_cst", "prc_reg_bd_cst", "n_sct_dft_cst", "n_seg_dft_cst", "n_reg_dft_cst", "n_sct_bd_cst", "n_seg_bd_cst", "n_reg_bd_cst"]
        "opacct_scoreout": load_module_scoreout("opacct"),
        "fin_scoreout": load_module_scoreout("fin"),
        "loan_scoreout": load_module_scoreout("loan"),
        "rev_scoreout": load_module_scoreout("rev"),
        "str": str_date,  # Dynamically calculated start date
        "end": end_date,  # Dynamically calculated end date
        "woe_code_hvr": woe_code_hvr,  # Function for WOE transformation
        "target_score": 200,
        "target_odds": 50,
        "pts_double_odds": 20
    }
    return config_data


# 2. Macro-like Functions for Processing Steps


def merge_macro(input1, input2, var, var_rename, keyA, keyB):
    """
    Simulates the merge_macro in SAS. Performs a left join between input1 and input2 on specified keys.
    
    Parameters:
    - input1 (pd.DataFrame): Left DataFrame.
    - input2 (pd.DataFrame): Right DataFrame.
    - var (str): Column name in input2 to select and rename.
    - var_rename (str): New column name for the selected variable.
    - keyA (str): Join key column name in input1.
    - keyB (str): Join key column name in input2.
    
    Returns:
    - pd.DataFrame: The resulting DataFrame after the left join.
    """
    try:
        # Select only the required column from input2 and rename it
        input2_renamed = input2[[keyB, var]].rename(columns={var: var_rename})
        
        # Perform the left join on the specified keys
        result = pd.merge(input1, input2_renamed, left_on=keyA, right_on=keyB, how="left")
        
        # Drop the duplicate key column from input2 after the join
        result = result.drop(columns=[keyB])
        
        return result
    except Exception as e:
        logging.error("Error in merge_macro: %s", e)
        raise

import pandas as pd
import pandasql as ps
import logging

def merge_macro(input1, input2, var, var_rename, keyA, keyB):
    """
    Performs a left join between two DataFrames using SQL syntax, similar to the SAS merge_macro.
    
    Parameters:
    - input1 (pd.DataFrame): Left DataFrame.
    - input2 (pd.DataFrame): Right DataFrame.
    - var (str): Column name in input2 to select and rename.
    - var_rename (str): New column name for the selected variable.
    - keyA (str): Join key column name in input1.
    - keyB (str): Join key column name in input2.
    
    Returns:
    - pd.DataFrame: The resulting DataFrame after the left join.
    """
    try:
        logging.info("Performing SQL-based left join using merge_macro.")
        
        # Renaming columns in input2 to avoid conflicts in SQL
        input2 = input2.rename(columns={var: var_rename})

        # SQL query for the left join
        query = f"""
        SELECT a.*, b.{var_rename}
        FROM input1 a
        LEFT JOIN input2 b
        ON a.{keyA} = b.{keyB}
        """
        
        # Execute the SQL query
        result = ps.sqldf(query, locals())
        
        return result
    
    except Exception as e:
        logging.error("Error in merge_macro: %s", e)
        raise



import pandas as pd
import logging

def dupu_en(datain):
    """
    Simulates the `%dupuen` macro in SAS, which performs deduplication with specific sorting criteria.

    Parameters:
    - datain (pd.DataFrame): The input DataFrame to be deduplicated.

    Returns:
    - pd.DataFrame: The deduplicated DataFrame.
    """
    try:
        logging.info("Starting dupu_en processing.")

        # Step 1: Copy input to avoid modifying the original DataFrame
        data_temp = datain.copy()

        # Step 2: Define `time_dim_key` and `rr`
        data_temp['time_dim_key'] = data_temp['RPT_PRD_END_DT']
        # Note: `rr` is skipped here as it’s commented out in the SAS code.
        # data_temp['rr'] = data_temp['grade_0']  # Uncomment if `rr` logic is needed

        # Step 3: Sort by `rel_uen`, `time_dim_key` in descending order of `rr`
        # If `rr` is required in future, uncomment and include it in sort order
        data_temp = data_temp.sort_values(by=['rel_uen', 'time_dim_key'], ascending=[True, False])

        # Step 4: Deduplicate by `rel_uen` and `time_dim_key`
        # This removes duplicates based on `rel_uen` and `time_dim_key`, keeping the first occurrence
        data_temp = data_temp.drop_duplicates(subset=['rel_uen', 'time_dim_key'])

        # Step 5: Drop `rr` and `time_dim_key` columns from the final output
        # Adjust as per requirements if `rr` needs to be calculated and kept
        data_temp = data_temp.drop(columns=['time_dim_key'], errors='ignore')

        logging.info("Completed dupu_en processing.")

        return data_temp

    except Exception as e:
        logging.error("Error in dupu_en: %s", e)
        raise


def portfolio_CI(df):
    """
    Simulates the `%portfolio_CI` macro in SAS by applying conditional logic to set the `CI` column.

    Parameters:
    - df (pd.DataFrame): The DataFrame to which the CI logic will be applied.

    Returns:
    - pd.DataFrame: The DataFrame with the `CI` column added.
    """
    try:
        logging.info("Applying portfolio_CI logic.")
        
        # Define the conditions for setting CI to 1
        lvl1_conditions = [
            'BMO CAPITAL MARKETS',
            'CANADIAN COMMERCIAL BANKING',
            'P&C US BUSINESS BANKING',
            'P&C US COMMERCIAL',
            'WEALTH MANAGEMENT',
            'HEALTH MANAGEMENT'
        ]
        
        rel_rsk_conditions = [
            'GC', 'LC', 'HGC', 'AVREMEDIA', 'HEALTHCM', 'HEALTHUS', 'HPB', 'RELIGIOU'
        ]
        
        # Apply conditional logic to set `CI` column
        df['CI'] = 0  # Default to 0
        df.loc[
            (df['LVL1_RPT_BOD_NM'].isin(lvl1_conditions)) &
            (df['REL_RSK_RTG_MODL_CD'].isin(rel_rsk_conditions)) &
            (df['ccb_uen_ind'] == 1),
            'CI'
        ] = 1
        
        logging.info("portfolio_CI logic applied successfully.")
        
        return df
    
    except Exception as e:
        logging.error("Error in portfolio_CI: %s", e)
        raise

def calpred_cmbn(data, mod_list):
    try:
        logging.info("Combining predictions from modules: %s", mod_list)
        # Placeholder for combined prediction calculation logic
        return data
    except Exception as e:
        logging.error("Failed during calpred_cmbn: %s", e)
        raise

import pandas as pd
import numpy as np
import logging

def cal_score(datain, predict_col, score_col, target_score=200, target_odds=50, pts_double_odds=20):
    """
    Calculates the score based on the predicted probability.

    Parameters:
    - datain (pd.DataFrame): Input DataFrame containing the prediction column.
    - predict_col (str): Column name containing the predicted probability.
    - score_col (str): Name of the output column for the calculated score.
    - target_score (float): Target score value (default 200).
    - target_odds (float): Target odds value (default 50).
    - pts_double_odds (float): Points to double the odds (default 20).

    Returns:
    - pd.DataFrame: DataFrame with the new score column.
    """
    try:
        logging.info("Calculating scores based on the predicted probability.")

        # Calculate factor and offset based on the given parameters
        factor = pts_double_odds / np.log(2)
        offset = target_score - factor * np.log(target_odds)

        # Calculate the score
        datain[score_col] = offset + factor * np.log((1 - datain[predict_col]) / datain[predict_col])

        logging.info("Score calculation completed successfully.")
        return datain

    except Exception as e:
        logging.error("Error in cal_score: %s", e)
        raise


import pandas as pd
import numpy as np
import logging

def sel_var_full(input_df, list_df, ind, output_name="output"):
    """
    Translates the SAS macro `sel_var_full` to Python, performing a set of transformations and calculations.

    Parameters:
    - input_df (pd.DataFrame): The input DataFrame to use for calculations.
    - list_df (pd.DataFrame): The list DataFrame to filter and transpose.
    - ind (int): The index to filter on in `list_df`.
    - output_name (str): The name of the output DataFrame.

    Returns:
    - pd.DataFrame: The transformed DataFrame with calculated predictions.
    """
    try:
        logging.info("Starting sel_var_full processing.")

        # Step 1: Filter `list_df` where `ind` matches
        tmp_lst = list_df[list_df['ind'] == ind][['Estimate']]

        # Step 2: Transpose `tmp_lst`
        tmp_lst_trns = tmp_lst.T
        tmp_lst_trns.columns = [f"var{i+1}" for i in range(tmp_lst_trns.shape[1])]

        # Step 3: Dynamically create columns with parameter values
        cnt = tmp_lst.shape[0]  # Number of matching records
        prm_columns = [f"prm{i+1}" for i in range(cnt)]

        # Flatten transposed values to use as parameters
        params = tmp_lst_trns.iloc[0].values  # Array of parameter values

        # Step 4: Calculate logistic regression (logit) and predictions
        # Start with a zeroed-out logit
        input_df['logit'] = 0

        # For each parameter, add the respective transformation to the logit
        for i in range(cnt):
            input_df['logit'] += params[i] * input_df[f"var{i+1}"]

        # Calculate predicted probability using logistic function
        input_df['predict'] = 1 / (1 + np.exp(-input_df['logit']))
        input_df = input_df.rename(columns={'predict': 'predict1'})

        # Optional: Drop intermediate columns (logit, var1, var2, etc.) if desired
        input_df.drop(columns=['logit'], inplace=True)

        logging.info("Completed sel_var_full processing.")
        
        # Rename output for clarity
        globals()[output_name] = input_df
        return input_df

    except Exception as e:
        logging.error("Error in sel_var_full: %s", e)
        raise

# # Example usage
# # Sample `input_df` with placeholder values for `var1`, `var2`, etc.
# input_df = pd.DataFrame({
#     'var1': [0.5, 0.7, 0.9],
#     'var2': [1.2, 0.8, 1.0]
# })

# # Sample `list_df` with `Estimate` and `ind` columns
# list_df = pd.DataFrame({
#     'Estimate': [0.3, 0.6],
#     'ind': [1, 1]
# })

# # Call the function with example data
# output_df = sel_var_full(input_df, list_df, ind=1)
# print(output_df)

import pandas as pd

def excl_list(df):
    """
    Exclude records based on specified conditions, equivalent to the `%excl_list` macro in SAS.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to apply exclusions.

    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """
    # Apply the conditions for exclusion
    df = df[df['Incl1'] == 1]
    df = df[df['LVL1_RPT_BOD_NM'] != 'WEALTH MANAGEMENT']
    df = df[df['LVL3_RPT_BOD_NM'] != 'PC-US FINANCIAL INSTITUTIONS']
    df = df[~df['REL_RSK_RTG_MODL_CD'].isin(['AVERMEDIA', 'HEALTHCM', 'HEALTHUS'])]
    
    return df

# # Example usage
# # Assuming `data` is a DataFrame containing the columns `Incl1`, `LVL1_RPT_BOD_NM`, `LVL3_RPT_BOD_NM`, and `REL_RSK_RTG_MODL_CD`
# data = pd.DataFrame({
#     'Incl1': [1, 0, 1],
#     'LVL1_RPT_BOD_NM': ['WEALTH MANAGEMENT', 'OTHER', 'OTHER'],
#     'LVL3_RPT_BOD_NM': ['PC-US FINANCIAL INSTITUTIONS', 'OTHER', 'OTHER'],
#     'REL_RSK_RTG_MODL_CD': ['AVERMEDIA', 'OTHER', 'HEALTHCM']
# })

# filtered_data = excl_list(data)
# print(filtered_data)

def model_excl(df):
    """
    Exclude records based on `TRN` value and then apply `excl_list` exclusions.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """
    # Exclude records where TRN is not 1
    df = df[df['TRN'] == 1]
    
    # Apply excl_list exclusions
    df = excl_list(df)
    
    return df

# # Example usage
# data['TRN'] = [1, 0, 1]
# filtered_data = model_excl(data)
# print(filtered_data)

def seg_ind_none(df):
    """
    Applies exclusions equivalent to `%seg_ind_none` by calling `excl_list`.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """
    return excl_list(df)

# # Example usage
# filtered_data_none = seg_ind_none(data)
# print(filtered_data_none)

import pandas as pd
import logging


def avg_trend(df, var_list):
    # Placeholder for average trend calculation
    return df


def coall(df, var_list, lag):
    """
    Python equivalent of the `coall` SAS macro. This function iterates over the `var_list` columns
    in the `df` DataFrame, creating new columns with values computed based on a lag parameter.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns.
    - var_list (list of str): List of column names to process.
    - lag (int): The lag parameter for the computation.

    Returns:
    - pd.DataFrame: The DataFrame with added columns based on `coall` logic.
    """
    for i, var in enumerate(var_list):
        if i > 0:
            # Create a new column by applying the coalesce operation with the lag
            df[f'{var}_coall'] = df[var].fillna(df[var_list[i - 1]] - lag)
        else:
            df[f'{var}_coall'] = df[var]  # First column remains unchanged
    return df

def trend(df, var_list):
    """
    Python equivalent of the `trend` SAS macro. Computes trends based on the `var_list`.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns.
    - var_list (list of str): List of column names to process.

    Returns:
    - pd.DataFrame: The DataFrame with added trend columns based on `trend` logic.
    """
    for i, var in enumerate(var_list):
        if i > 0:
            # Compute trends using various divisions of consecutive variables
            if (df[var_list[i - 1]] != 0).all() and (df[var] != 0).all():
                df[f'm_{var}'] = df[var_list[i - 1]] / df[var]
            df[f'q_{var}'] = df[var_list[i - 1]] / df[var] if (df[var] != 0).all() else None
            df[f's_{var}'] = df[var_list[i - 1]] / df[var] if (df[var] != 0).all() else None
            df[f'y_{var}'] = df[var_list[i - 1]] / df[var] if (df[var] != 0).all() else None
    return df

def abs_trend(df, var_list):
    """
    Python equivalent of the `abs_trend` SAS macro. Computes absolute trends for `var_list`.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns.
    - var_list (list of str): List of column names to process.

    Returns:
    - pd.DataFrame: The DataFrame with added absolute trend columns.
    """
    for i, var in enumerate(var_list):
        if i > 0:
            # Compute absolute trends using various operations
            df[f'am_{var}'] = abs(df[var_list[i - 1]] - df[var])
            df[f'aq_{var}'] = abs(df[var_list[i - 1]] - df[var])
            df[f'as_{var}'] = abs(df[var_list[i - 1]] - df[var])
            df[f'ay_{var}'] = abs(df[var_list[i - 1]] - df[var])
    return df

# # Example usage
# # Assuming `data` is a DataFrame with columns in `var_list`
# data = pd.DataFrame({
#     'var1': [10, 20, 30, 40],
#     'var2': [15, 25, 35, 45],
#     'var3': [5, 10, 15, 20]
# })
# var_list = ['var1', 'var2', 'var3']
# lag = 5

# # Applying coall, trend, and abs_trend functions
# data = coall(data, var_list, lag)
# data = trend(data, var_list)
# data = abs_trend(data, var_list)

# print(data)

import pandas as pd
import numpy as np

def avg_trend(df, var_list):
    """
    Python equivalent of the `avg_trend` SAS macro. Computes average, sum, max, and relative changes
    over 3-month and 6-month periods for columns in `var_list`.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns.
    - var_list (list of str): List of column names to process.

    Returns:
    - pd.DataFrame: The DataFrame with added average trend columns based on `avg_trend` logic.
    """
    for var in var_list:
        # Calculate 3 and 6 month averages
        df[f'ag3_{var}'] = df[var].shift(1).rolling(window=3).mean()
        df[f'ag6_{var}'] = df[var].shift(1).rolling(window=6).mean()
        
        # Calculate 3 and 6 month sums
        df[f'sum3_{var}'] = df[var].shift(1).rolling(window=3).sum()
        df[f'sum6_{var}'] = df[var].shift(1).rolling(window=6).sum()
        
        # Calculate 3 and 6 month maximums
        df[f'max3_{var}'] = df[var].shift(1).rolling(window=3).max()
        df[f'max6_{var}'] = df[var].shift(1).rolling(window=6).max()

        # Compute relative change compared to same period last year: 3-month and 6-month
        if df[var].shift(13).notnull().any():
            df[f'crm3_{var}'] = (df[f'sum3_{var}'] - df[var].shift(13).rolling(window=3).sum()) / df[var].shift(13).rolling(window=3).sum()
            df[f'crm6_{var}'] = (df[f'sum6_{var}'] - df[var].shift(13).rolling(window=6).sum()) / df[var].shift(13).rolling(window=6).sum()
            
            # Handle cases where the rolling sum returns NaN
            df[f'crm3_{var}'] = df[f'crm3_{var}'].replace([np.inf, -np.inf], np.nan).fillna(0)
            df[f'crm6_{var}'] = df[f'crm6_{var}'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Calculate relative change in average compared to same period last year: 3-month and 6-month
        if df[var].shift(13).notnull().any():
            df[f'cr3_{var}'] = (df[f'ag3_{var}'] - df[var].shift(13).rolling(window=3).mean()) / df[var].shift(13).rolling(window=3).mean()
            df[f'cr6_{var}'] = (df[f'ag6_{var}'] - df[var].shift(13).rolling(window=6).mean()) / df[var].shift(13).rolling(window=6).mean()
            
            # Handle cases where the rolling mean returns NaN
            df[f'cr3_{var}'] = df[f'cr3_{var}'].replace([np.inf, -np.inf], np.nan).fillna(0)
            df[f'cr6_{var}'] = df[f'cr6_{var}'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df

# # Example usage
# # Assuming `data` is a DataFrame with columns in `var_list`
# data = pd.DataFrame({
#     'var1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
#     'var2': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195]
# })
# var_list = ['var1', 'var2']

# # Apply avg_trend function
# data = avg_trend(data, var_list)

# print(data)

import pandas as pd
import pandasql as ps

def all_out_scorer_no_seg1(scoreout, mod_data, cfg):
    """
    Python equivalent of the `all_out_scorer_no_seg1` SAS macro.
    
    Parameters:
    - scoreout (pd.DataFrame): Input DataFrame, assumed to be similar to `scoreout` in SAS.
    - mod_data (pd.DataFrame): The module data to be joined.
    - cfg (dict): Configuration dictionary containing necessary parameters and lists of variables.

    Returns:
    - pd.DataFrame: The transformed DataFrame with the scoring logic applied.
    """
    
    # Step 1: SQL join to create `temp1` (equivalent to proc sql join in SAS)
    query = """
    SELECT a.*, b.uen_ID AS mod_rel_uen, b.*
    FROM scoreout a
    INNER JOIN mod_data b ON a.rpt_prd_end_dt = b.rpt_prd_end_dt AND a.rel_uen = b.uen_ID
    """
    temp1 = ps.sqldf(query, locals())

    # Step 2: Apply macros on `temp1`
    # Assuming cfg contains 'coal', 'rel', and 'abs' variables
    temp1 = coall(temp1, cfg["coal"], lag=1)
    temp1 = trend(temp1, cfg["rel"])
    temp1 = abs_trend(temp1, cfg["abs"])
    temp1 = avg_trend(temp1, cfg["abs"])

    # Step 3: Include the external WOE code if needed (assuming `woe_code_<mod>` is a function)
    # This would dynamically call a function or apply transformations specific to WOE code
    woe_code_func = cfg.get("woe_code_func")
    if callable(woe_code_func):
        temp2 = woe_code_func(temp1)
    else:
        temp2 = temp1  # If no WOE code is provided, proceed with temp1 as temp2

    # Step 4: Apply `sel_var_full` with the list and index (simulating variable selection and renaming)
    # `multi_<mod>` and `combo_<mod>` need to be defined in cfg or passed separately
    temp3 = sel_var_full(temp2, cfg["multi_mod"], cfg["combo_mod"])

    # Step 5: Score calculation (equivalent to `%cal_score` macro in SAS)
    temp3 = cal_score(temp3, target_score=cfg["target_score"], target_odds=cfg["target_odds"], pts_double_odds=cfg["pts_double_odds"])

    # Rename the `predict` and `score` fields based on module
    mod = cfg["mod"]
    temp3 = temp3.rename(columns={"score": f"score_{mod}", "predict": f"predict_{mod}"})

    # Step 6: Concatenate segments if needed (following `tmpout` structure from SAS)
    tmpout = temp3.copy()

    return tmpout

# Placeholder function for sel_var_full, assuming it is a column selector/renamer
def sel_var_full(df, list_var, ind):
    # Placeholder for the actual logic of sel_var_full
    # Implement the necessary selection and renaming as per SAS macro logic
    return df

# Example usage
# Assume scoreout and mod_data are DataFrames loaded with appropriate data, and `cfg` is defined
cfg = config()  # Load the configuration
result_df = all_out_scorer_no_seg1(scoreout, mod_data, cfg)
print(result_df)



# Function to dynamically rename columns (replaces `%cust_gen_rnm` macro)
def cust_gen_rnm(df):
    for i in range(1, 19):
        df.rename(columns={
            f'percent_ind_sect_default_cust{i}': f'prc_sct_dft_cst{i}',
            f'percent_ind_seg_default_cust{i}': f'prc_seg_dft_cst{i}',
            f'percent_region_default_cust{i}': f'prc_reg_dft_cst{i}',
            f'percent_ind_sect_bad_cust{i}': f'prc_sct_bd_cst{i}',
            f'percent_ind_seg_bad_cust{i}': f'prc_seg_bd_cst{i}',
            f'percent_region_bad_cust{i}': f'prc_reg_bd_cst{i}',
            f'n_ind_sect_default_cust{i}': f'n_sct_dft_cst{i}',
            f'n_ind_seg_default_cust{i}': f'n_seg_dft_cst{i}',
            f'n_region_default_cust{i}': f'n_reg_dft_cst{i}',
            f'n_ind_sect_bad_cust{i}': f'n_sct_bd_cst{i}',
            f'n_ind_seg_bad_cust{i}': f'n_seg_bd_cst{i}',
            f'n_region_bad_cust{i}': f'n_reg_bd_cst{i}'
        }, inplace=True)
    return df

import pandas as pd
import numpy as np

def op_acct_trans_spec(df):
    """
    Equivalent to SAS `op_acct_trans_spec` macro.
    """
    df['turnover_to_credit1'] = df[['sum_credit_trans1', 'sum_credit_trans3']].mean(axis=1) / df['balance1'].replace({0: 1})
    df['turnover_to_credit2'] = df[['sum_credit_trans1', 'sum_credit_trans6']].mean(axis=1) / df['balance2'].replace({0: 1})
    df['turnover_to_credit3'] = df[['sum_credit_trans1', 'sum_credit_trans9']].mean(axis=1) / df['balance3'].replace({0: 1})

    df['cr_trns_vol1'] = df[['sum_credit_trans1', 'sum_credit_trans3']].max(axis=1) - df[['sum_credit_trans1', 'sum_credit_trans3']].min(axis=1)
    df['cr_trns_vol2'] = df[['sum_credit_trans1', 'sum_credit_trans6']].max(axis=1) - df[['sum_credit_trans1', 'sum_credit_trans6']].min(axis=1)
    df['cr_trns_vol3'] = df[['sum_credit_trans1', 'sum_credit_trans9']].max(axis=1) - df[['sum_credit_trans1', 'sum_credit_trans9']].min(axis=1)

    df['distance_to_usage1'] = df['balance1'] / (df[['balance1', 'sum_credit_trans1', 'min_balance1']].max(axis=1) + 0.0001)
    df['distance_to_usage2'] = df['balance2'] / (df[['balance2', 'sum_credit_trans2', 'min_balance2']].max(axis=1) + 0.0001)
    df['distance_to_usage3'] = df['balance3'] / (df[['balance3', 'sum_credit_trans3', 'min_balance3']].max(axis=1) + 0.0001)
    
    return df

def op_acct_trans_lag(df):
    """
    Equivalent to SAS `op_acct_trans_lag` macro.
    """
    for i in range(1, 19):
        df[f'free_limit{i}'] = df[[f'balance{i}', f'limit{i}']].max(axis=1)
        df[f'shr_f_bggst_cr_trns{i}'] = df[f'free_limit{i}'].fillna(0) / df[f'sum_credit_trans{i}'].replace({0: 1})
    return df



# Example usage
# Assuming `df` is your initial DataFrame with necessary columns and date range defined
# str_date and end_date need to be defined
# df = merge_opacct(df)
# print(df)


# 3. Processing Steps

def filter_entity_data(data, start_date, end_date):
    try:
        logging.info("Filtering entity data within date range.")
        query = f"""
        SELECT * FROM data
        WHERE rpt_prd_end_dt >= '{start_date}' AND rpt_prd_end_dt <= '{end_date}'
        """
        return ps.sqldf(query, locals())
    except Exception as e:
        logging.error("Failed during filter_entity_data: %s", e)
        raise

def load_hvr_data(hvr_data, woe_code_func):
    try:
        logging.info("Applying WOE logic to HVR data.")
        return woe_code_func(hvr_data)
    except Exception as e:
        logging.error("Failed during load_hvr_data: %s", e)
        raise

def merge_hvr_and_entity(all_out_time1, hvr_arm):
    try:
        logging.info("Merging HVR and entity data.")
        query = """
        SELECT a.*, b.cg6UNR, b.woe_cg6UNR, b.grp_cg6UNR
        FROM all_out_time1 a
        LEFT JOIN hvr_arm b
        ON a.rpt_prd_end_dt = b.rpt_prd_end_dt AND a.rel_uen = b.rel_uen
        """
        return ps.sqldf(query, locals())
    except Exception as e:
        logging.error("Failed during merge_hvr_and_entity: %s", e)
        raise

def generate_cust_gen_scoreout(cfg):

    # Step 1: Load and filter `cust_gen_input` by date and rename columns
    rolledup_customer_information = cfg["cust_gen_scoreout"][
        (cfg["cust_gen_scoreout"]['rpt_prd_end_dt'] >= cfg["str"]) & 
        (cfg["cust_gen_scoreout"]['rpt_prd_end_dt'] <= cfg["end"])
    ].copy()
    
    # Rename columns according to SAS code requirements
    rolledup_customer_information['n_dep_prod'] = rolledup_customer_information['# of deposit product']
    rolledup_customer_information['n_lend_prod'] = np.nan
    rolledup_customer_information['Yrs_in_Bus_old'] = np.nan
    rolledup_customer_information['Yrs_w_Bank_old'] = np.nan

    # Apply dynamic renaming as per `cust_gen_rnm`
    rolledup_customer_information = cust_gen_rnm(rolledup_customer_information)

    # Step 2: Process `fn_entity` to create `c1`
    c1 = cfg["fn_entity"][(cfg["fn_entity"]['uen'].notna()) & (cfg["fn_entity"]['ACTV_IND'] == 'Y')].copy()
    c1['client_id'] = np.where(c1['cctn_uen'].notna(), c1['cctn_uen'], c1['uen'])
    c1['bad'] = np.where(
        (c1['mstr_scl_rtg_cd'].str[0].isin(['P', 'T', 'D'])) & (c1['ACTV_BRWR_IND'] == 'Y'), 
        1, 
        0
    )
    c1 = c1[['rpt_prd_end_dt', 'client_id', 'rel_uen', 'uen', 'bad']]

    # Step 3: Aggregate counts in `c1` to create `c2`
    c2 = c1[(c1['rpt_prd_end_dt'] >= cfg["str"]) & (c1['rpt_prd_end_dt'] <= cfg["end"])]
    c2 = c2.groupby(['rpt_prd_end_dt', 'client_id']).agg(
        Connection_Count=('client_id', 'size'), 
        Connection_Bad=('bad', 'sum')
    ).reset_index()

    # Step 4: Deduplicate `c2` to create `countfinal`
    countfinal = c2.drop_duplicates(subset=['rpt_prd_end_dt', 'client_id'])

    # Step 5: Join `rolledup_customer_information` with `countfinal`
    rolledup_c1_sort_w_connect4 = rolledup_customer_information.merge(
        countfinal, left_on=['MONTH_CAPTURED', 'REL_UEN'], right_on=['rpt_prd_end_dt', 'rel_uen'], how='left'
    )
    rolledup_c1_sort_w_connect4 = rolledup_c1_sort_w_connect4[
        (rolledup_c1_sort_w_connect4['MONTH_CAPTURED'] >= cfg["str"]) & 
        (rolledup_c1_sort_w_connect4['MONTH_CAPTURED'] <= cfg["end"])
    ]

    # Placeholder for `%all_out_scorer_no_seg1` SAS macro logic
    # This would include further processing on `rolledup_c1_sort_w_connect4`
    tmpout = rolledup_c1_sort_w_connect4.copy()

    # Step 6: Set the final output DataFrame with module-specific naming
    scoreout = tmpout.copy()
    scoreout.columns = [f"{col}_{cfg['mod']}" if col in ['score', 'predict'] else col for col in scoreout.columns]

    logging.info("Scoreout generation process completed.")
    
    return scoreout

def generate_opacct_scoreout (df):
    """
    Main function to merge and process opacct data.
    """
    df = op_acct_trans_spec(df)
    df = op_acct_trans_lag(df)
    
    # Filter based on date range (assumes `rpt_prd_end_dt`, `str_date`, and `end_date` are in datetime format)
    df = df[(df['rpt_prd_end_dt'] >= str_date) & (df['rpt_prd_end_dt'] <= end_date)]
    
    return df

import pandas as pd
import numpy as np

def create_scoreout_comb(scoreout, cust_gen_scoreout, hvr_scoreout, opacct_scoreout, fin_scoreout, loan_scoreout, rev_scoreout):
    """
    Equivalent to the SQL join and data processing in the SAS code.
    
    Parameters:
    - scoreout (pd.DataFrame): The main DataFrame with initial scoring data.
    - cust_gen_scoreout, hvr_scoreout, opacct_scoreout, fin_scoreout, loan_scoreout, rev_scoreout (pd.DataFrame):
      DataFrames with additional scores for each module.

    Returns:
    - pd.DataFrame: A combined DataFrame after joins and additional processing.
    """
    # Step 1: Perform left joins on `rel_uen` and `rpt_prd_end_dt` columns
    scoreout_comb = scoreout \
        .merge(cust_gen_scoreout[['rel_uen', 'rpt_prd_end_dt', 'score', 'predict']], on=['rel_uen', 'rpt_prd_end_dt'], how='left', suffixes=('', '_cust_gen')) \
        .merge(hvr_scoreout[['rel_uen', 'rpt_prd_end_dt', 'score', 'predict']], on=['rel_uen', 'rpt_prd_end_dt'], how='left', suffixes=('', '_hvr')) \
        .merge(opacct_scoreout[['rel_uen', 'rpt_prd_end_dt', 'score', 'predict']], on=['rel_uen', 'rpt_prd_end_dt'], how='left', suffixes=('', '_opacct')) \
        .merge(fin_scoreout[['rel_uen', 'rpt_prd_end_dt', 'score', 'predict']], on=['rel_uen', 'rpt_prd_end_dt'], how='left', suffixes=('', '_fin')) \
        .merge(loan_scoreout[['rel_uen', 'rpt_prd_end_dt', 'score', 'predict']], on=['rel_uen', 'rpt_prd_end_dt'], how='left', suffixes=('', '_loan')) \
        .merge(rev_scoreout[['rel_uen', 'rpt_prd_end_dt', 'score', 'predict']], on=['rel_uen', 'rpt_prd_end_dt'], how='left', suffixes=('', '_rev'))
    
    # Step 2: Rename prediction columns for clarity
    scoreout_comb = scoreout_comb.rename(columns={
        'predict_cust_gen': 'pred_cust_gen_hvr',
        'predict_opacct': 'pred_op_acct',
        'predict_fin': 'pred_fin',
        'predict_loan': 'pred_loan',
        'predict_rev': 'pred_rev'
    })

    # Step 3: Define `piece` categories based on prediction column presence
    conditions = [
        (scoreout_comb['pred_cust_gen_hvr'].notna() & scoreout_comb['pred_fin'].notna() & 
         scoreout_comb['pred_op_acct'].isna() & scoreout_comb['pred_loan'].isna() & scoreout_comb['pred_rev'].isna()),
        
        (scoreout_comb['pred_op_acct'].notna() | scoreout_comb['pred_loan'].notna() | scoreout_comb['pred_rev'].notna()),
        
        (scoreout_comb['pred_cust_gen_hvr'].notna() & scoreout_comb['pred_fin'].isna() & 
         scoreout_comb['pred_op_acct'].isna() & scoreout_comb['pred_loan'].isna() & scoreout_comb['pred_rev'].isna()),
        
        (scoreout_comb['pred_cust_gen_hvr'].isna() & scoreout_comb['pred_fin'].isna() & 
         scoreout_comb['pred_op_acct'].isna() & scoreout_comb['pred_loan'].isna() & scoreout_comb['pred_rev'].isna())
    ]
    choices = [1, 2, 3, 4]
    scoreout_comb['piece'] = np.select(conditions, choices, default=np.nan)

    # Step 4: Assign values to X_ and M_ columns based on predictions
    scoreout_comb['X_cg'] = scoreout_comb['pred_cust_gen_hvr'].fillna(0)
    scoreout_comb['M_cg'] = np.where(scoreout_comb['pred_cust_gen_hvr'].isna(), 1, 0)

    scoreout_comb['X_fin'] = scoreout_comb['pred_fin'].fillna(0)
    scoreout_comb['M_fin'] = np.where(scoreout_comb['pred_fin'].isna(), 1, 0)

    scoreout_comb['X_oa'] = scoreout_comb['pred_op_acct'].fillna(0)
    scoreout_comb['M_oa'] = np.where(scoreout_comb['pred_op_acct'].isna(), 1, 0)

    scoreout_comb['X_loan'] = scoreout_comb['pred_loan'].fillna(0)
    scoreout_comb['M_loan'] = np.where(scoreout_comb['pred_loan'].isna(), 1, 0)

    scoreout_comb['X_rev'] = scoreout_comb['pred_rev'].fillna(0)
    scoreout_comb['M_rev'] = np.where(scoreout_comb['pred_rev'].isna(), 1, 0)

    return scoreout_comb

# # Example usage
# # Assuming `scoreout`, `cust_gen_scoreout`, `hvr_scoreout`, `opacct_scoreout`, `fin_scoreout`, `loan_scoreout`, and `rev_scoreout`
# # are already defined and contain the appropriate data

# result_df = create_scoreout_comb(scoreout, cust_gen_scoreout, hvr_scoreout, opacct_scoreout, fin_scoreout, loan_scoreout, rev_scoreout)
# print(result_df)





# def assign_piece(data):
#     try:
#         logging.info("Assigning piece values based on prediction combinations.")
#         conditions = [
#             (~data['pred_cust_gen_hvr'].isnull() & ~data['pred_fin'].isnull() & data['pred_op_acct'].isnull() & data['pred_loan'].isnull() & data['pred_rev'].isnull()),
#             (data['pred_op_acct'].notnull() | data['pred_loan'].notnull() | data['pred_rev'].notnull()),
#             (~data['pred_cust_gen_hvr'].isnull() & data['pred_fin'].isnull() & data['pred_op_acct'].isnull() & data['pred_loan'].isnull() & data['pred_rev'].isnull()),
#             (data['pred_cust_gen_hvr'].isnull() & data['pred_fin'].isnull() & data['pred_op_acct'].isnull() & data['pred_loan'].isnull() & data['pred_rev'].isnull())
#         ]
#         choices = [1, 2, 3, 4]
#         data['piece'] = np.select(conditions, choices, default=np.nan)
#         return data
#     except Exception as e:
#         logging.error("Failed during assign_piece: %s", e)
#         raise



def process_scoreout_comb(scoreout_comb):
    """
    Process scoreout_comb DataFrame to calculate integrated scores based on module combinations and scale final scores.
    
    Parameters:
    - scoreout_comb (pd.DataFrame): Combined scoreout DataFrame after initial joins and processing.
    
    Returns:
    - pd.DataFrame: Final processed DataFrame with integrated and scaled scores.
    """
    
    # Step 1: Calculate integrated scores based on module combinations
    # Define mod_list and mod_list_name for the first combination
    mod_list = ['cust_gen_hvr', 'fin']
    mod_list_name = 'cg_f'
    
    # Filter where piece == 1 and apply calpred_cmbn logic (this is a placeholder for the actual implementation)
    scoreout_comb1 = scoreout_comb[scoreout_comb['piece'] == 1].copy()
    scoreout_comb1['pred'] = calpred_cmbn(scoreout_comb1, mod_list, mod_list_name)
    
    # For piece == 3 and piece == 4, handle specific conditions for 'pred'
    scoreout_comb1_piece3 = scoreout_comb[scoreout_comb['piece'] == 3].copy()
    scoreout_comb1_piece3['pred'] = scoreout_comb1_piece3['pred_cust_gen_hvr'].combine_first(-np.log(1 + 1 / 0.0462))
    
    scoreout_comb1_piece4 = scoreout_comb[scoreout_comb['piece'] == 4].copy()
    scoreout_comb1_piece4['pred'] = scoreout_comb1_piece4['pred_fin']
    
    # Concatenate the results for piece 1, 3, and 4 into scoreout_comb1
    scoreout_comb1 = pd.concat([scoreout_comb1, scoreout_comb1_piece3, scoreout_comb1_piece4], ignore_index=True)
    
    # Step 2: Calculate integrated scores based on a different set of modules for piece == 2
    mod_list = ['X_cg', 'X_fin', 'X_oa', 'X_loan', 'X_rev', 'M_fin', 'M_oa', 'M_loan', 'M_rev']
    mod_list_name = 'cg_f_oa_l_r'
    
    scoreout_comb2 = scoreout_comb[scoreout_comb['piece'] == 2].copy()
    scoreout_comb2['pred'] = calpred_cmbn2(scoreout_comb2, mod_list, mod_list_name)
    
    # Step 3: Combine scoreout_comb1 and scoreout_comb2 into final intg_scoreout
    intg_scoreout = pd.concat([scoreout_comb1, scoreout_comb2], ignore_index=True)
    intg_scoreout['pred1'] = 1 / (1 + np.exp(-intg_scoreout['pred']))
    
    # Step 4: Scale the score
    intg_scoreout1 = cal_score(intg_scoreout, 'pred1', 'intg_score', target_score=200, target_odds=50, pts_double_odds=20)
    
    return intg_scoreout1

def calpred_cmbn(df, mod_list, param_lib, pe_name):
    """
    Equivalent to SAS `calpred_cmbn` macro. Calculates predictions using parameters from param_lib.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with relevant columns.
    - mod_list (list of str): List of modules to consider in prediction.
    - param_lib (pd.DataFrame): DataFrame containing model parameters.
    - pe_name (str): Name to identify the parameter set in param_lib.

    Returns:
    - pd.Series: Calculated prediction values.
    """
    # Initialize the prediction with the intercept value
    intercept = param_lib[(param_lib['pe_name'] == pe_name) & (param_lib['Variable'] == 'Intercept')]['Estimate'].values[0]
    logit = intercept

    # For each variable in mod_list, add its weighted estimate to the logit
    for var in mod_list:
        estimate = param_lib[(param_lib['pe_name'] == pe_name) & (param_lib['Variable'] == f"pred_{var}")]['Estimate'].values[0]
        logit += df[var] * estimate

    # Calculate and return the predicted value
    pred = 1 / (1 + np.exp(-logit))
    return pred

def calpred_cmbn2(df, mod_list, param_lib, pe_name):
    """
    Equivalent to SAS `calpred_cmbn2` macro. Similar to `calpred_cmbn`, but uses a different approach.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with relevant columns.
    - mod_list (list of str): List of modules to consider in prediction.
    - param_lib (pd.DataFrame): DataFrame containing model parameters.
    - pe_name (str): Name to identify the parameter set in param_lib.

    Returns:
    - pd.Series: Calculated prediction values.
    """
    # Initialize the prediction with the intercept value
    intercept = param_lib[(param_lib['pe_name'] == pe_name) & (param_lib['Variable'] == 'Intercept')]['Estimate'].values[0]
    logit = intercept

    # For each variable in mod_list, add its weighted estimate to the logit
    for var in mod_list:
        estimate = param_lib[(param_lib['pe_name'] == pe_name) & (param_lib['Variable'] == var)]['Estimate'].values[0]
        logit += df[var] * estimate

    # Calculate and return the predicted value
    pred = 1 / (1 + np.exp(-logit))
    return pred

# # Example usage
# # Assuming `df` is the input DataFrame, `param_lib` contains parameters, and mod_list is defined
# # Replace `mod_list`, `param_lib`, and `pe_name` with actual values as needed
# mod_list = ["X_cg", "X_fin", "X_oa", "X_loan", "X_rev"]
# pe_name = "cg_f"
# param_lib = pd.DataFrame({
#     'pe_name': ["cg_f", "cg_f", "cg_f", "cg_f", "cg_f"],
#     'Variable': ["Intercept", "X_cg", "X_fin", "X_oa", "X_loan"],
#     'Estimate': [1.5, 0.3, -0.4, 0.2, 0.1]
# })

# # Calculate predictions
# df['pred'] = calpred_cmbn(df, mod_list, param_lib, pe_name)
# df['pred2'] = calpred_cmbn2(df, mod_list, param_lib, pe_name)


# Example usage
# Assuming `scoreout_comb` is the DataFrame with the combined scores and necessary columns
final_scoreout = process_scoreout_comb(scoreout_comb)
print(final_scoreout)


# 4. Main Pipeline

def main_pipeline():
    try:
        logging.info("Starting main pipeline.")
        
        # Load configuration and inputs
        cfg = config()
        
        # Step 1: Filter Entity Data
        all_out_time1 = filter_entity_data(cfg["fn_entity"], cfg["str"], cfg["end"])
        
        # Step 2: Load HVR Data and Apply WOE Logic
        hvr_arm = load_hvr_data(cfg["hvr_input"], cfg["apply_woe_transformation"])
        
        # Step 3: Merge HVR and Entity Data
        all_out_time = merge_hvr_and_entity(all_out_time1, hvr_arm)
        
        # Step 4: Merge with MSTR_SCL and Assign Grades
        all_out_time2 = merge_macro(all_out_time, cfg["MSTR_SCL"], "grade", "grade_0", "MSTR_SCL_RTG_CD", "MSTR_SCL_RTG_CD")
        
        # Step 5: Filter Based on Portfolio CI and Deduplicate
        # Apply initial filter for ACTV_BRWR_IND == 'Y'
        all_out_time2 = all_out_time2[all_out_time2['ACTV_BRWR_IND'] == 'Y']
        
        # Apply deduplication using dupu_en function (equivalent to `%dupuen`)
        all_out_time2 = dupu_en(all_out_time2)

        # Create `scoreout` dataset with portfolio CI and additional filters
        scoreout = portfolio_CI(all_out_time2)  # Apply portfolio CI function

        # Apply additional filters and `incl1` logic
        scoreout['incl1'] = scoreout['grade_0'].apply(lambda x: 0 if x > 11 else 1)
        scoreout = scoreout[(scoreout['CI'] != 0) &  # CI should not be zero
                            (~scoreout['grade_0'].isna())]  # Remove rows where grade_0 is NaN

        # Step 6: Final Deduplication for Scoreout
        scoreout = dupu_en(scoreout)  # Deduplicate `scoreout` to ensure clean final data
        
        logging.info("Pipeline completed successfully.")
        
        return scoreout

    except Exception as e:
        logging.error("Failed in main pipeline: %s", e)
        raise

# Run the pipeline
final_result = main_pipeline()
print(final_result)



