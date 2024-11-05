import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Read the SAS file
def read_sas_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

# Identify and extract macros, PROC steps, and data dependencies
def extract_elements(sas_lines):
    macros = {}
    procs = []
    data_dependencies = []
    
    macro_name = None
    in_macro = False
    for line in sas_lines:
        # Identify macro definition
        macro_def = re.match(r"%macro\s+(\w+)", line, re.IGNORECASE)
        if macro_def:
            macro_name = macro_def.group(1)
            macros[macro_name] = []
            in_macro = True

        # Identify end of macro
        if in_macro and re.search(r"%mend", line, re.IGNORECASE):
            macro_name = None
            in_macro = False

        # Identify macro calls
        if in_macro and macro_name:
            macro_call = re.findall(r"%(\w+)\s*\(", line)
            macros[macro_name].extend(macro_call)
        
        # Identify PROC steps
        proc_step = re.match(r"proc\s+(\w+)", line, re.IGNORECASE)
        if proc_step:
            procs.append(proc_step.group(1))
        
        # Identify dataset dependencies
        data_step = re.search(r"data\s*=\s*(\w+)", line, re.IGNORECASE)
        if data_step:
            data_dependencies.append(data_step.group(1))
    
    return macros, procs, data_dependencies

# Build a process flow graph
def build_graph(macros, procs, data_dependencies):
    graph = nx.DiGraph()

    # Add macros and their calls
    for macro, calls in macros.items():
        graph.add_node(macro, type='macro')
        for call in calls:
            graph.add_node(call, type='macro')  # Ensure called macros have a type
            graph.add_edge(macro, call)

    # Add PROC steps as nodes
    for proc in procs:
        graph.add_node(proc, type='proc')

    # Add data dependencies as nodes and edges
    for i in range(len(data_dependencies) - 1):
        graph.add_node(data_dependencies[i], type='dataset')  # Default type for datasets
        graph.add_node(data_dependencies[i + 1], type='dataset')
        graph.add_edge(data_dependencies[i], data_dependencies[i + 1], type='data_dependency')
    
    return graph

# Visualize the graph
def visualize_and_save_graph(graph, output_path="procedure_flow.png"):
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(12, 12))

    # Color nodes by type with a default color for nodes without a type
    colors = [
        'skyblue' if data.get('type') == 'macro' else 'lightgreen' if data.get('type') == 'proc' else 'lightcoral'
        for _, data in graph.nodes(data=True)
    ]
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=2000)
    nx.draw_networkx_edges(graph, pos, arrowstyle='->', arrowsize=20)
    nx.draw_networkx_labels(graph, pos, font_size=10)

    plt.title("SAS Process Flow Graph")
    # Save the plot as a high-resolution PNG
    plt.savefig(output_path, format="png", dpi=300)  # 300 DPI for high resolution
    plt.show()

# Main function to execute the analysis
def main(sas_file_path):
    sas_lines = read_sas_file(sas_file_path)
    macros, procs, data_dependencies = extract_elements(sas_lines)
    graph = build_graph(macros, procs, data_dependencies)
    visualize_and_save_graph(graph)

# Run the analysis
sas_file_path = 'complex_sas.sas'  # Update with the actual path to your SAS file
main(sas_file_path)
