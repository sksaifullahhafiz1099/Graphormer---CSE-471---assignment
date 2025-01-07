# Graphormer---CSE-471---Assignment

## Introduction
Transformers have transformed fields like NLP and computer vision, but they’ve struggled to match Graph Neural Networks (GNNs) on graph representation benchmarks. This raises a key question: Are Transformers unsuitable for graph data, or is there a better way to use them?

The paper *Graphormer* answers this by introducing methods to effectively encode graph structure into Transformers, enabling state-of-the-art performance on graph tasks. Key innovations include:

- **Centrality Encoding**: Captures node importance using degree centrality.
- **Spatial Encoding**: Encodes relationships between nodes using shortest path distances.
- **Edge Encoding**: Incorporates edge-specific features like bond types in molecular graphs.

With these enhancements, Graphormer bridges the gap between GNNs and Transformers, achieving exceptional results on tasks like molecular property prediction and large-scale challenges (e.g., OGB-LSC). 

This blog explores how Graphormer works, the challenges it addresses, and its potential to reshape graph representation learning.

---

## Message Passing and Aggregation in Graph Neural Networks (GNNs)
GNNs are powerful tools for learning from graph-structured data. They rely on **message passing and aggregation** to propagate information through the graph. Here’s an overview:

### 1. Understanding Graphs
A graph comprises nodes (entities) and edges (relationships). Nodes and edges may have features that add context, such as user attributes in social networks or bond types in molecules.

### 2. Message Passing
Message passing is the core operation in GNNs. Each node \(i\) gathers information from its neighbors:

- **Message Generation**: Neighbors send their features to \(i\).  
- **Message Aggregation**: Node \(i\) combines the received messages.

Mathematically:
```math
h_i^{(l+1)} = \text{Aggregate} \big( \{ f(h_i^{(l)}, h_j^{(l)}, e_{ij}) \, | \, j \in \mathcal{N}(i) \} \big)
```
### Where:
- $h_i^{(l)}$: Feature of node $i$ at layer $l$.  
- $e_{ij}$: Feature of the edge between $i$ and $j$.  
- $\mathcal{N}(i)$: Set of neighbors of $i$.  
- $f$: Function to compute messages.  
- **Aggregate**: Combines messages (e.g., sum, average).  

---

### 3. Aggregation
Nodes aggregate messages to update their features, capturing both local structure and context. Common methods include summing, averaging, or using attention mechanisms.

---

### 4. Stacking Layers
By stacking multiple layers, GNNs allow nodes to gather information from farther graph regions, enabling multi-hop relationships.

---

### 5. Challenges of GNNs
Despite their success, GNNs face limitations:
- **Over-smoothing**: Node features become indistinguishable with deeper layers.  
- **Local Focus**: Struggle to capture global graph properties.  
- **Scalability**: Computationally expensive for large graphs.  

---

### Why Graphormer?  
While GNNs excel at local structure, their global modeling is limited. Graphormer overcomes these challenges by integrating graph structure directly into the Transformer architecture, blending the strengths of both approaches.

Next, we’ll explore how Graphormer achieves this!


