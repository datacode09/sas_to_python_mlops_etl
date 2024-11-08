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


