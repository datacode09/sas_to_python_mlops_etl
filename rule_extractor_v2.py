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

def extract_rules_from_section(variable_text):
    """Extract individual rules from a variable section text."""
    rules = []
    capturing_rule = False
    rule_lines = []

    for line in variable_text:
        line = line.strip()
        if line.startswith("LABEL"):
            # Skip LABEL lines as they are for documentation only
            continue

        if line.startswith("IF"):
            # Start capturing a new rule block
            capturing_rule = True
            rule_lines = [line]
        elif capturing_rule:
            # Continue capturing the current rule block until "END;"
            rule_lines.append(line)
            if "END;" in line:
                capturing_rule = False
                # Join the lines and add the complete rule to the list
                rules.append(" ".join(rule_lines))

    print(f"Extracted rules: {rules}")  # Debug output
    return rules

def process_intermediate_data(intermediate_data):
    """Process each variable in the intermediate dataset to extract rules."""
    variable_rules = {}
    for var, section_text in intermediate_data.items():
        print(f"Processing variable: '{var}'")
        rules = extract_rules_from_section(section_text)
        variable_rules[var] = rules

    print("Final extracted rules for all variables:", json.dumps(variable_rules, indent=2))  # Debug output
    return variable_rules

# Test content
test_content = [
    "* Variable: Accounts_Payable_Days;",
    "LABEL GRP_Accounts_Payable_Days = 'Grouped: Accounts_Payable_Days';",
    "LABEL WOE_Accounts_Payable_Days = 'Weight of Evidence: Accounts_Payable_Days';",
    "IF MISSING(Accounts_Payable_Days) THEN DO;",
    "    GRP_Accounts_Payable_Days = 5;",
    "    WOE_Accounts_Payable_Days = -0.38022322927;",
    "END;",
    "IF NOT MISSING(Accounts_Payable_Days) THEN DO;",
    "    IF Accounts_Payable_Days < 5.96 THEN DO;",
    "        GRP_Accounts_Payable_Days = 1;",
    "        WOE_Accounts_Payable_Days = 0.1326085862;",
    "    END;",
    "END;",
    "* Variable: Accounts_Payable_Days1;",
    "LABEL GRP_Accounts_Payable_Days1 = 'Grouped: Accounts_Payable_Days1';",
    "LABEL WOE_Accounts_Payable_Days1 = 'Weight of Evidence: Accounts_Payable_Days1';",
    "IF MISSING(Accounts_Payable_Days1) THEN DO;",
    "    GRP_Accounts_Payable_Days1 = 5;",
    "    WOE_Accounts_Payable_Days1 = -0.38022322927;",
    "END;",
    "IF NOT MISSING(Accounts_Payable_Days1) THEN DO;",
    "    IF Accounts_Payable_Days1 < 5.96 THEN DO;",
    "        GRP_Accounts_Payable_Days1 = 1;",
    "        WOE_Accounts_Payable_Days1 = 0.1326085862;",
    "    END;",
    "    IF Accounts_Payable_Days1 >= 5.96 AND Accounts_Payable_Days1 < 73.06 THEN DO;",
    "        GRP_Accounts_Payable_Days1 = 2;",
    "        WOE_Accounts_Payable_Days1 = 0.5422631185;",
    "    END;",
    "END;"
]

# Run the functions
intermediate_data = create_intermediate_dataset(test_content)
processed_rules = process_intermediate_data(intermediate_data)
