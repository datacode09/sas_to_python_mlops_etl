# Macro Complexity and Cyclomatic Complexity Analysis
import re
from collections import defaultdict

# Read SAS file
def read_sas_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

# Calculate macro complexity
def calculate_macro_complexity(sas_lines):
    macro_complexity = {}
    in_macro = False
    current_macro = None
    complexity_count = 0

    for line in sas_lines:
        # Detect macro start and end
        macro_def = re.match(r"%macro\s+(\w+)", line, re.IGNORECASE)
        if macro_def:
            current_macro = macro_def.group(1)
            complexity_count = 0
            in_macro = True
        elif re.search(r"%mend", line, re.IGNORECASE):
            if current_macro:
                macro_complexity[current_macro] = complexity_count
                current_macro = None
            in_macro = False

        # Count complexity factors within macros
        if in_macro:
            complexity_count += len(re.findall(r"%if|%do|%while|%else", line, re.IGNORECASE))

    return macro_complexity

# Main function
def main(file_path):
    sas_lines = read_sas_file(file_path)
    complexity = calculate_macro_complexity(sas_lines)
    print("Macro Complexity (Cyclomatic Complexity):")
    for macro, comp in complexity.items():
        print(f"{macro}: {comp}")

# Run analysis
file_path = 'path/to/your/sasfile.sas'  # Update with your file path
main(file_path)

