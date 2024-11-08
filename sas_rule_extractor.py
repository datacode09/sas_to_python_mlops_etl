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
    """Extract rules for each variable from the SAS file content, formatted with '* Variable:'."""
    variable_rules = {}
    current_var = None
    current_rules = []

    try:
        for i, line in enumerate(content):
            # Display each line for detailed debugging
            print(f"Processing line {i+1}: '{line.strip()}'")  

            # Trim whitespace and match "* Variable:" case-insensitively, with any surrounding spaces
            line = line.strip()
            if re.search(r"(?i)^\s*\* variable\s*:", line):  # Match lines like "* Variable: VARIABLE_NAME;"
                # Save previous variable's rules if any
                if current_var is not None and current_rules:
                    variable_rules[current_var] = current_rules
                    print(f"Saved rules for variable: '{current_var}'")  # Confirm saving rules

                # Extract the variable name flexibly
                parts = line.split(":")
                if len(parts) > 1:
                    current_var = parts[1].strip()  # Capture the variable name after "* Variable:"
                    current_rules = []
                    print(f"Detected new variable: '{current_var}'")  # Debugging output for detected variables
                else:
                    print(f"Warning: Variable declaration found but no name on line {i+1}")

            elif line.startswith("LABEL"):
                # Skip LABEL lines as they're for documentation
                print(f"Skipping LABEL line on line {i+1}")

            elif line.startswith("IF") and current_var:  # Start of an IF condition block
                # Collect the entire IF block as a rule
                rule_lines = [line]
                for j in range(i+1, len(content)):
                    if "END;" in content[j]:  # Look for the end of the IF block
                        rule_lines.append(content[j].strip())
                        break
                    rule_lines.append(content[j].strip())
                current_rules.append(" ".join(rule_lines))
                print(f"Captured rule for variable '{current_var}' on line {i+1}")

        # Save the last variable's rules
        if current_var is not None and current_rules:
            variable_rules[current_var] = current_rules
            print(f"Saved rules for last variable: '{current_var}'")

        if not variable_rules:
            raise ValueError("No rules were found in the content.")
        
        print("Variable rules extracted successfully.")
        print("Extracted Rules:", variable_rules)  # Debugging output
        return variable_rules

    except Exception as e:
        print(f"An error occurred while extracting variable rules on line {i+1}: {e}")
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
