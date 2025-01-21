import matplotlib.pyplot as plt
import networkx as nx

# Create directed graph for the Hebbian model
G = nx.DiGraph()

# Add nodes for inputs, neurons, and outputs
G.add_nodes_from(["Input1", "Input2", "Input3", "Input4", "Input5", "Input6", "Input7", "Input8",
                  "Neuron1", "Neuron2", "Neuron3", "Output1", "Output2", "Output3"])

# Add edges from inputs to neurons and neurons to outputs
edges = []
for input_node in ["Input1", "Input2", "Input3", "Input4", "Input5", "Input6", "Input7", "Input8"]:
    edges.extend([(input_node, "Neuron1"), (input_node, "Neuron2"), (input_node, "Neuron3")])

edges.extend([
    ("Neuron1", "Output1"), ("Neuron1", "Output2"), ("Neuron1", "Output3"),
    ("Neuron2", "Output1"), ("Neuron2", "Output2"), ("Neuron2", "Output3"),
    ("Neuron3", "Output1"), ("Neuron3", "Output2"), ("Neuron3", "Output3")
])

G.add_edges_from(edges)

# Define positions for the graph layout
pos = {
    **{f"Input{i+1}": (0, 8-i) for i in range(8)},  # Inputs arranged vertically
    "Neuron1": (2, 6), "Neuron2": (2, 4), "Neuron3": (2, 2),  # Neurons horizontally aligned
    "Output1": (4, 5), "Output2": (4, 3), "Output3": (4, 1)   # Outputs horizontally aligned
}

# Plot the graph
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold", edge_color="gray")

# Add edge labels (all weights set to 1 for simplicity)
edge_labels = {edge: "" for edge in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="black", font_weight="bold")

# Add title to the graph
plt.title("Hebbian Neural Network Diagram", fontsize=14)
plt.show()
