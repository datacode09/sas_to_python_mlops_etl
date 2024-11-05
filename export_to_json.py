import json
import networkx as nx
from networkx.readwrite import json_graph

# Function to export graph to JSON format
def export_graph_to_json(graph, output_path="graph.json"):
    # Convert the graph to a JSON-compatible format
    data = json_graph.node_link_data(graph)  # Converts to node-link format
    # Write the data to a JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Graph exported to {output_path} successfully.")

# Example usage with your existing graph
# Assume dataset_graph and proc_graph are created from previous code

# Export dataset flow graph to JSON
export_graph_to_json(dataset_graph, output_path="dataset_flow.json")

# Export procedure flow graph to JSON
export_graph_to_json(proc_graph, output_path="procedure_flow.json")
