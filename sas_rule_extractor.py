import pandas as pd
import numpy as np
import re

def load_rules(file_path):
    """Load and parse the rules from a SAS file."""
    try:
        with open(file_path, 'r') as file:
            content = file.readlines()
        if not content:
            raise ValueError("The rules file is empty.")
        print("Rules file loaded successfully.")
        return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred while loading the rules file: {e}")
        return []

def extract_variable_rules(content):
    """Extract rules for each variable from the SAS file content."""
    variable_rules = {}
    current_var = None
    current_rules = []

    try:
        for line in content:
            line = line.strip()
            if line.startswith("Variable:"):
                # Save the previous variable and its rules
                if current_var is not None:
                    variable_rules[current_var] = current_rules

                # Start a new variable
                current_var = line.split(":")[1].strip()
                current_rules = []
            elif line:
                # Collect rule lines for the current variable
                current_rules.append(line)

        # Save the last variable's rules
        if current_var is not None:
            variable_rules[current_var] = current_rules

        if not variable_rules:
            raise ValueError("No rules were found in the content.")
        
        print("Variable rules extracted successfully.")
        print("Extracted Rules:", variable_rules)  # Debugging output
        return variable_rules

    except Exception as e:
        print(f"An error occurred while extracting variable rules: {e}")
        return {}

def parse_rule(line):
    """Parse an individual rule line to condition and assignment."""
    try:
        # Pattern to match IF-THEN or ELSE statements
        match_if = re.match(r'IF (.+?) THEN (.+);', line, re.IGNORECASE)
        match_else = re.match(r'ELSE (.+);', line, re.IGNORECASE)

        if match_if:
            condition = match_if.group(1).strip()
            assignment = match_if.group(2).strip()
        elif match_else:
            condition = 'True'  # ELSE applies unconditionally
            assignment = match_else.group(1).strip()
        else:
            raise ValueError(f"Invalid rule format: {line}")

        print("Parsed Rule - Condition:", condition, "| Assignment:", assignment)  # Debugging output
        return condition, assignment

    except ValueError as ve:
        print(f"Value Error: {ve}")
        return None, None
    except Exception as e:
        print(f"An error occurred while parsing the rule '{line}': {e}")
        return None, None

def apply_variable_rules(df, variable, rules):
    """Apply rules for a specific variable to the DataFrame."""
    for line in rules:
        condition, assignment = parse_rule(line)
        if condition and assignment:
            try:
                # Extract column and value from assignment (e.g., "GRP_VAR = 1")
                column, value = re.match(r"(\w+) = (.+)", assignment).groups()
                # Check if the column exists in the DataFrame; if not, initialize it
                if column not in df.columns:
                    df[column] = np.nan

                # Evaluate the condition and apply the assignment
                print(f"Applying Rule - Condition: {condition}, Assignment: {column} = {value}")  # Debugging output
                df.loc[df.eval(condition), column] = eval(value) if "'" not in value else value.strip("'")

            except AttributeError:
                print(f"Attribute Error: Invalid assignment format in rule '{line}' for variable '{variable}'")
            except ValueError as ve:
                print(f"Value Error: {ve} in rule '{line}' for variable '{variable}'")
            except Exception as e:
                print(f"An error occurred while applying rule '{line}' for variable '{variable}': {e}")
    return df

def apply_all_rules(df, variable_rules):
    """Apply rules for all variables to the DataFrame."""
    try:
        if not variable_rules:
            raise ValueError("No variable rules to apply.")
        
        for variable, rules in variable_rules.items():
            print(f"Applying rules for variable: {variable}")
            df = apply_variable_rules(df, variable, rules)

        print("All rules applied successfully.")
        return df

    except ValueError as ve:
        print(f"Value Error: {ve}")
    except Exception as e:
        print(f"An error occurred while applying all rules: {e}")
    return df

# Main script execution
if __name__ == "__main__":
    # Path to the SAS rules file
    sas_file_path = 'rules.sas'  # Replace with the path to your .sas file
    
    # Load and parse the rules file
    content = load_rules(sas_file_path)
    if not content:
        print("Exiting due to failed rule loading.")
    else:
        variable_rules = extract_variable_rules(content)

        # Sample DataFrame for demonstration - Replace with your actual data
        data = {
            'BORROWER_RISK_RATING': [41, 43, 45, 46, 47, np.nan],
            'CONNECTED_AUTHORIZATION': [400000, 500000, 5300000, 10000000, 100, np.nan]
        }
        df = pd.DataFrame(data)

        # Apply all rules
        df = apply_all_rules(df, variable_rules)
        
        # Print the updated DataFrame
        print("Updated DataFrame:\n", df)
