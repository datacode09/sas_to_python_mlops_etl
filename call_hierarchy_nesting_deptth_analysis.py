import re
import networkx as nx
import matplotlib.pyplot as plt

def extract_macro_hierarchy(sas_lines):
    hierarchy = defaultdict(list)
    current_macro = None
    in_macro = False

    for line in sas_lines:
        macro_def = re.match(r"%macro\s+(\w+)", line, re.IGNORECASE)
        if macro_def:
            current_macro = macro_def.group(1)
            in_macro = True
        elif re.search(r"%mend", line, re.IGNORECASE):
            current_macro = None
            in_macro = False

        # Capture macro calls within other macros
        if in_macro and current_macro:
            called_macros = re.findall(r"%(\w+)\(", line)
            hierarchy[current_macro].extend(called_macros)

    return hierarchy

def visualize_hierarchy(hierarchy):
    graph = nx.DiGraph()
    for macro, calls in hierarchy.items():
        for call in calls:
            graph.add_edge(macro, call)
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx(graph, pos, node_size=2000, node_color='lightblue', arrows=True)
    plt.title("Macro Call Hierarchy")
    plt.show()

def main(file_path):
    sas_lines = read_sas_file(file_path)
    hierarchy = extract_macro_hierarchy(sas_lines)
    visualize_hierarchy(hierarchy)

file_path = 'path/to/your/sasfile.sas'
main(file_path)
