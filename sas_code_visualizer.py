import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Enhanced visualization function
def visualize_enhanced_graph(graph):
    pos = nx.spring_layout(graph, seed=42)  # Consistent layout with a fixed seed
    plt.figure(figsize=(14, 10))

    # Define node types and colors
    color_map = {'macro': 'skyblue', 'proc': 'lightgreen', 'dataset': 'salmon'}
    shape_map = {'macro': 'o', 'proc': 's', 'dataset': 'D'}
    labels = nx.get_node_attributes(graph, 'type')
    
    # Draw nodes with different shapes and colors based on their type
    for node_type, color in color_map.items():
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=[node for node, attr in graph.nodes(data=True) if attr.get('type') == node_type],
            node_shape=shape_map[node_type],
            node_color=color,
            label=node_type.capitalize(),
            node_size=2000,
            edgecolors="black"
        )

    # Draw edges with different styles for macro calls vs. data dependencies
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), arrows=True, arrowstyle='->', arrowsize=20)

    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_size=10)

    # Add legend (only color-based since shapes can't be displayed in legend directly)
    legend_elements = [
        Patch(facecolor=color_map['macro'], edgecolor='black', label='Macro'),
        Patch(facecolor=color_map['proc'], edgecolor='black', label='Procedure'),
        Patch(facecolor=color_map['dataset'], edgecolor='black', label='Dataset'),
    ]
    plt.legend(handles=legend_elements, loc='best')

    plt.title("Enhanced SAS Process Flow Graph")
    plt.axis('off')
    plt.show()

    
    
# Assuming all the previous functions (e.g., read_sas_file, extract_elements, build_graph) are defined

# Read the SAS file and build the graph
sas_file_path = 'complex_sas.sas'  # Update with the actual path to your SAS file
sas_lines = read_sas_file(sas_file_path)       # Read the SAS file
macros, procs, data_dependencies = extract_elements(sas_lines)  # Extract elements
graph = build_graph(macros, procs, data_dependencies)           # Build the graph

# Now visualize the graph with the enhanced visualization function

# Assuming the `graph` object is already created
visualize_enhanced_graph(graph)
