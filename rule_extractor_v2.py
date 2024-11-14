import json
import re

def create_intermediate_dataset(content):
    """Scan the text and create an intermediate JSON-like structure with variable sections."""
    intermediate_data = {}
    current_var = None
    current_text = []

    for line in content:
        line = line.strip()
        if re.search(r"(?i)^\s*\* variable\s*:", line):  # Detect "* Variable:" line
            # If there's an active variable section, save it before moving to the next
            if current_var is not None:
                intermediate_data[current_var] = current_text
                current_text = []

            # Extract variable name after "* Variable:"
            parts = line.split(":")
            current_var = parts[1].strip() if len(parts) > 1 else None
            print(f"Detected new variable: '{current_var}'")

        elif current_var:  # Collect text under the current variable section
            current_text.append(line)

    # Save the last variable section
    if current_var is not None:
        intermediate_data[current_var] = current_text

    print("Intermediate data structure:", json.dumps(intermediate_data, indent=2))  # Debug output
    return intermediate_data


import re

def extract_rules_from_section(variable_text):
    """Extract individual rules from a SAS variable section text, treating keywords as case-insensitive."""
    rules = []
    capturing_rule = False
    rule_lines = []

    for line in variable_text:
        line = line.strip()

        # Skip LABEL lines as they are for documentation only
        if re.match(r"^LABEL", line, re.IGNORECASE):
            continue

        # Start capturing a rule block when we encounter an IF statement with "THEN DO;"
        if re.match(r"^IF .+ THEN DO;", line, re.IGNORECASE):
            capturing_rule = True
            rule_lines = [line]
            print(f"Started capturing rule at line: '{line}'")  # Debug output
        elif capturing_rule:
            # Continue capturing lines for the current rule until "END;" is found
            rule_lines.append(line)
            if re.search(r"END;", line, re.IGNORECASE):  # Case-insensitive check for END;
                capturing_rule = False
                # Join the lines and add the complete rule to the list
                complete_rule = " ".join(rule_lines)
                rules.append(complete_rule)
                print(f"Completed rule capture: '{complete_rule}'")  # Debug output
                rule_lines = []  # Reset for the next rule

    print(f"Extracted rules: {rules}")  # Debug output
    return rules

def process_intermediate_data(intermediate_data):
    """Process each variable in the intermediate dataset to extract rules."""
    variable_rules = {}
    for var, section_text in intermediate_data.items():
        print(f"Processing variable: '{var}'")  # Debug output
        rules = extract_rules_from_section(section_text)
        variable_rules[var] = rules

    print("Final extracted rules for all variables:", json.dumps(variable_rules, indent=2))  # Debug output
    return variable_rules


def parse_sas_condition(condition):
    """Convert a SAS condition to a Python-compatible condition, handling variables with underscores."""
    # Replace `MISSING(variable)` with `pd.isna(df['variable'])`
    condition = re.sub(r"MISSING\((\w+)\)", r"pd.isna(df['\1'])", condition, flags=re.IGNORECASE)

    # Replace `variable eq value` with `df['variable'] == value`, supporting underscore variables
    condition = re.sub(r"(\w+)\s+eq\s+(['\w\d\.]+)", r"df['\1'] == \2", condition, flags=re.IGNORECASE)
    
    # Replace `variable ne value` with `df['variable'] != value`, supporting underscore variables
    condition = re.sub(r"(\w+)\s+ne\s+(['\w\d\.]+)", r"df['\1'] != \2", condition, flags=re.IGNORECASE)

    # Replace `AND` and `OR` with `&` and `|`
    condition = condition.replace("AND", "&").replace("OR", "|")

    return condition


def apply_rule_to_dataframe(df, rule):
    """Apply a single rule block with nested IF, ELSE IF, and ELSE conditions."""
    # Split the rule into individual condition-action pairs
    conditions = re.findall(r"(IF\s+.+?\s+THEN\s+DO;|ELSE\s+IF\s+.+?\s+THEN\s+DO;|ELSE\s+DO;)", rule, re.IGNORECASE)
    actions = re.split(r"IF\s+.+?\s+THEN\s+DO;|ELSE\s+IF\s+.+?\s+THEN\s+DO;|ELSE\s+DO;", rule, re.IGNORECASE)[1:]
    
    for condition, action_block in zip(conditions, actions):
        # Determine the condition to apply, handling "IF", "ELSE IF", and "ELSE"
        if condition.startswith("IF") or condition.startswith("ELSE IF"):
            condition_str = re.search(r"IF (.+?) THEN DO;", condition, re.IGNORECASE).group(1)
            parsed_condition = parse_sas_condition(condition_str)
            mask = eval(parsed_condition)
        else:  # ELSE case, apply where no previous conditions matched
            mask = ~df.loc[df.index].any(axis=1)  # Apply to rows where no previous conditions matched
        
        # Apply each assignment in the action block for rows that match the mask
        for assignment in action_block.split(";"):
            assignment = assignment.strip()
            if assignment:
                target_column, value = assignment.split("=")
                target_column = target_column.strip()
                value = value.strip()
                
                # Ensure the target column exists
                if target_column not in df.columns:
                    df[target_column] = None
                
                # Apply assignment based on the mask
                df.loc[mask, target_column] = eval(value)
                print(f"Applied rule: {target_column} = {value} where {parsed_condition if condition.startswith('IF') else 'ELSE'}")


