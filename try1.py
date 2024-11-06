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


import pandas as pd

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


def load_module_scoreout(module_name):
    try:
        logging.info("Loading module scoreout data for %s.", module_name)
        return pd.DataFrame()  # Example placeholder
    except Exception as e:
        logging.error("Failed to load module scoreout data for %s: %s", module_name, e)
        raise

def woe_code_hvr(df):
    try:
        logging.info("Applying WOE code transformations.")
        # Placeholder for WOE transformation logic
        return df
    except Exception as e:
        logging.error("Failed to apply WOE code transformations: %s", e)
        raise

# 2. Configuration Function to Gather Inputs

from datetime import datetime, timedelta
import pandas as pd

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


def config():
    logging.info("Configuring inputs.")
    try:
        config_data = {
            "fn_entity": load_fn_entity(),
            "hvr_input": load_hvr_input(),
            "MSTR_SCL": load_mstr_scl(),
            "cust_gen_scoreout": load_module_scoreout("cust_gen"),
            "opacct_scoreout": load_module_scoreout("opacct"),
            "fin_scoreout": load_module_scoreout("fin"),
            "loan_scoreout": load_module_scoreout("loan"),
            "rev_scoreout": load_module_scoreout("rev"),
            "str": "2024-01-01",
            "end": "2024-12-31",
            "woe_code_hvr": woe_code_hvr,
            "target_score": 200,
            "target_odds": 50,
            "pts_double_odds": 20
        }
        return config_data
    except Exception as e:
        logging.error("Failed to configure inputs: %s", e)
        raise

# 2. Macro-like Functions for Processing Steps

import pandas as pd

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

def cal_score(data, target_score, target_odds, pts_double_odds):
    try:
        logging.info("Calculating scaled score.")
        data['intg_score'] = data['pred1'] * target_score / (target_score + target_odds + pts_double_odds)
        return data
    except Exception as e:
        logging.error("Failed during cal_score: %s", e)
        raise

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

def assign_piece(data):
    try:
        logging.info("Assigning piece values based on prediction combinations.")
        conditions = [
            (~data['pred_cust_gen_hvr'].isnull() & ~data['pred_fin'].isnull() & data['pred_op_acct'].isnull() & data['pred_loan'].isnull() & data['pred_rev'].isnull()),
            (data['pred_op_acct'].notnull() | data['pred_loan'].notnull() | data['pred_rev'].notnull()),
            (~data['pred_cust_gen_hvr'].isnull() & data['pred_fin'].isnull() & data['pred_op_acct'].isnull() & data['pred_loan'].isnull() & data['pred_rev'].isnull()),
            (data['pred_cust_gen_hvr'].isnull() & data['pred_fin'].isnull() & data['pred_op_acct'].isnull() & data['pred_loan'].isnull() & data['pred_rev'].isnull())
        ]
        choices = [1, 2, 3, 4]
        data['piece'] = np.select(conditions, choices, default=np.nan)
        return data
    except Exception as e:
        logging.error("Failed during assign_piece: %s", e)
        raise

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



