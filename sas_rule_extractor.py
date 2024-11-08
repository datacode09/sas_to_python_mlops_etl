import pandas as pd
import numpy as np
import re

def load_rules(file_path):
    """Load and parse the rules from a SAS file."""
    with open(file_path, 'r') as file:
        content = file.readlines()
    return content

def extract_variable_rules(content):
    """Extract rules for each variable from the SAS file content."""
    variable_rules = {}
    current_var = None
    current_rules = []

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

    return variable_rules

def parse_rule(line):
    """Parse an individual rule line to condition and assignment."""
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
        condition = None
        assignment = None

    return condition, assignment

def apply_variable_rules(df, variable, rules):
    """Apply rules for a specific variable to the DataFrame."""
    for line in rules:
        condition, assignment = parse_rule(line)
        if condition and assignment:
            # Extract column and value from assignment (e.g., "GRP_VAR = 1")
            column, value = re.match(r"(\w+) = (.+)", assignment).groups()
            # Evaluate the condition and apply the assignment
            df.loc[df.eval(condition), column] = eval(value) if "'" not in value else value.strip("'")
    return df

def apply_all_rules(df, variable_rules):
    """Apply rules for all variables to the DataFrame."""
    for variable, rules in variable_rules.items():
        print(f"Applying rules for variable: {variable}")
        df = apply_variable_rules(df, variable, rules)
    return df

# Main script execution
if __name__ == "__main__":
    # Path to the SAS rules file
    sas_file_path = 'rules.sas'  # Replace with the path to your .sas file
    
    # Load and parse the rules file
    content = load_rules(sas_file_path)
    variable_rules = extract_variable_rules(content)

    # Sample DataFrame for demonstration - Replace with your actual data
    data = {
        'BORROWER_RISK_RATING': [41, 43, 45, 46, 47, np.nan],
        'CONNECTED_AUTHORIZATION': [400000, 500000, 5300000, 10000000, 100, np.nan]
    }
    df = pd.DataFrame(data)
    
    # Initialize columns mentioned in the rules with default values
    for variable in variable_rules.keys():
        for rule in variable_rules[variable]:
            _, assignment = parse_rule(rule)
            if assignment:
                column = assignment.split('=')[0].strip()
                if column not in df.columns:
                    df[column] = np.nan  # Add column if it doesn't exist in the DataFrame

    # Apply all rules
    df = apply_all_rules(df, variable_rules)
    
    # Print the updated DataFrame
    print(df)
