# dataiku_score_launcher.py
# Import required Dataiku libraries
import dataiku
from dataiku import Dataset, Folder
from datetime import datetime, timedelta
import pandas as pd

# Configuration class for Dataiku environment
class DaikuConfig:
    def __init__(self):
        # Adapt paths for Dataiku
        self.input_dataset = dataiku.Dataset("input_data")
        self.output_dataset = dataiku.Dataset("output_data") 
        self.gov_folder = dataiku.Folder("governance")
        self.alerts_dataset = dataiku.Dataset("alert_reasons")
        
        # Calculate dates similar to original SAS code
        today = datetime.today()
        self.start_dt = (today - timedelta(months=3)).strftime('%d%b%Y')
        self.end_dt = (today - timedelta(months=1)).strftime('%d%b%Y')
        self.run_from_dt = '31Dec2023'

class ScoreLauncher:
    def __init__(self, config):
        self.config = config
        self.model = 'CI'
        
    def read_input_data(self):
        """
        Read data from Dataiku dataset
        """
        return self.config.input_dataset.get_dataframe()

    def proc_means(self, df):
        """
        Equivalent to SAS PROC MEANS
        """
        summary_stats = df.groupby(['group']).agg({
            'value': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        return summary_stats

    def merge_models(self, df1, df2):
        """
        Model merging logic
        """
        return pd.merge(df1, df2, on='key', how='left')

class AlertProcessor:
    def __init__(self, config):
        self.config = config
        
    def suppress_alerts(self, alerts_df):
        """
        Alert suppression logic
        """
        suppression_rules = self.config.alerts_dataset.get_dataframe()
        merged = pd.merge(alerts_df, suppression_rules, on='alert_id', how='left')
        return merged[merged['suppress'] != True]

    def append_history(self, new_alerts):
        """
        Append to history dataset in Dataiku
        """
        history_dataset = dataiku.Dataset("alerts_history")
        if history_dataset.read_schema():
            history = history_dataset.get_dataframe()
            updated_history = pd.concat([history, new_alerts])
            # Write back to Dataiku dataset
            history_dataset.write_with_schema(updated_history)
        else:
            history_dataset.write_with_schema(new_alerts)

def main():
    # Initialize Dataiku configuration
    config = DaikuConfig()
    
    # Initialize processors
    scorer = ScoreLauncher(config)
    alert_processor = AlertProcessor(config)
    
    # Launch scoring process
    input_data = scorer.read_input_data()
    
    if not input_data.empty:
        # Process means
        means_results = scorer.proc_means(input_data)
        
        # Merge models
        model_results = scorer.merge_models(means_results, input_data)
        
        # Write results to Dataiku dataset
        config.output_dataset.write_with_schema(model_results)
        
        # Process alerts
        alerts = model_results[model_results['flag'] == 1].copy()
        suppressed_alerts = alert_processor.suppress_alerts(alerts)
        
        # Append to history
        alert_processor.append_history(suppressed_alerts)
        
        print("Processing completed successfully")

if __name__ == "__main__":
    main()
