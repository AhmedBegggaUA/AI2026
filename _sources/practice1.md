# Clique Problem

## Practical Session: Maximum Clique Problem

### Introduction

In previous sessions, we covered the fundamentals of message passing, dataset creation (Part A), and Graph Neural Networks (Part B). Now we will dive deep into solving a classic combinatorial optimization problem: the **Maximum Clique (MC) Problem**.

A **clique** in an undirected graph $G = (V, E)$ is a subset of vertices $C \subseteq V$ such that every pair of vertices in $C$ is connected by an edge. The **maximum clique** is the clique with the largest number of vertices. Finding the maximum clique is an NP-hard problem with applications in social network analysis, bioinformatics, pattern recognition, and knowledge discovery.

In this practical session, we will explore four approaches to solve the Maximum Clique problem, progressively building towards more sophisticated methods:

0. **Brute Force**: Exhaustive search through all possible subsets (baseline, only for small graphs)
1. **Motzkin-Straus Formulation**: Understanding how the discrete MC problem maps to continuous optimization
2. **Replicator Dynamics (RD)**: An iterative solver for the continuous formulation
3. **GNN + Replicator Dynamics**: A learned approach that improves RD initialization

We will apply these methods to the **IMDB** and **COLLAB** datasets from PyTorch Geometric.

### Prerequisites

```python
pip install torch torch_geometric networkx matplotlib numpy scikit-learn
```

### Datasets

We will use two graph classification datasets from TU Dortmund, available through PyTorch Geometric:

```python
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from itertools import combinations
import time

# Load IMDB-BINARY dataset
imdb_dataset = TUDataset(root='./data', name='IMDB-BINARY')
print(f"IMDB-BINARY: {len(imdb_dataset)} graphs")

# Load COLLAB dataset  
collab_dataset = TUDataset(root='./data', name='COLLAB')
print(f"COLLAB: {len(collab_dataset)} graphs")

# Examine a sample graph
sample_graph = imdb_dataset[0]
print(f"\nSample graph statistics:")
print(f"  Nodes: {sample_graph.num_nodes}")
print(f"  Edges: {sample_graph.num_edges}")
```

---

## Part 0: Brute Force Maximum Clique

### Why Start with Brute Force?

Before using sophisticated methods, we need a **ground truth** to validate our solutions. The brute force approach guarantees finding the exact maximum clique by checking all possible subsets of vertices.

### The Algorithm

The idea is simple:
1. Generate all possible subsets of vertices, starting from the largest
2. For each subset, check if it forms a clique (all pairs connected)
3. Return the first (largest) clique found

**Pseudocode:**

```
Algorithm: Brute Force Maximum Clique
─────────────────────────────────────
Input: Graph G = (V, E) with n vertices
Output: Maximum clique C*

1. For k = n down to 1:
   a. For each subset S ⊆ V of size k:
      i.  If Is_Clique(S):
          Return S
2. Return ∅  // Empty graph
```

**Helper function - Is_Clique:**

```
Function: Is_Clique(S, A)
─────────────────────────
Input: Subset S, Adjacency matrix A
Output: True if S is a clique, False otherwise

1. For each pair (i, j) in S where i < j:
   a. If A[i,j] = 0:
      Return False
2. Return True
```

### Complexity Analysis

- Number of subsets of size $k$: $\binom{n}{k}$
- Total subsets: $2^n$
- Checking if a subset of size $k$ is a clique: $O(k^2)$

**This is exponential!** For $n=30$, we have over 1 billion subsets. This is why we need smarter methods.

### Implementation Task

**Task 0.1:** Implement the helper function to check if a set of nodes forms a clique.

```python
def is_clique(nodes, A):
    """
    Check if a set of nodes forms a clique.
    
    Args:
        nodes: list of node indices
        A: numpy array of shape (n, n) - adjacency matrix
    
    Returns:
        bool: True if nodes form a clique, False otherwise
    """
    # TODO: Implement this function
    # Check that every pair of nodes in the list is connected
    pass
```

**Task 0.2:** Implement the brute force maximum clique algorithm.

```python
def brute_force_max_clique(A):
    """
    Find the maximum clique by brute force enumeration.
    
    Args:
        A: numpy array of shape (n, n) - adjacency matrix
    
    Returns:
        list: nodes forming the maximum clique
    """
    n = A.shape[0]
    
    # TODO: Implement brute force search
    # Start from largest possible clique size and work down
    # Use itertools.combinations to generate subsets
    pass
```

**Task 0.3:** Test on small graphs and measure the time.

```python
def test_brute_force():
    """
    Test brute force on small graphs and measure computation time.
    """
    # Create a simple test graph: triangle (clique of size 3) plus one extra node
    # Nodes 0, 1, 2 form a triangle; node 3 connects only to node 2
    A = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=float)
    
    start = time.time()
    clique = brute_force_max_clique(A)
    elapsed = time.time() - start
    
    print(f"Maximum clique: {clique}")
    print(f"Clique size: {len(clique)}")
    print(f"Time: {elapsed:.4f} seconds")
    
    # Verify it's actually a clique
    print(f"Is valid clique: {is_clique(clique, A)}")

test_brute_force()
```

**Task 0.4:** Apply brute force to IMDB and COLLAB datasets (first few small graphs only).

```python
def evaluate_brute_force_on_dataset(dataset, max_graphs=10, max_nodes=25):
    """
    Evaluate brute force on dataset (only small graphs).
    
    Args:
        dataset: PyTorch Geometric dataset
        max_graphs: maximum number of graphs to evaluate
        max_nodes: skip graphs larger than this (too slow)
    
    Returns:
        results: list of (graph_idx, clique_size, time) tuples
    """
    results = []
    
    for i in range(min(max_graphs, len(dataset))):
        data = dataset[i]
        
        if data.num_nodes > max_nodes:
            print(f"Graph {i}: Skipping (too large: {data.num_nodes} nodes)")
            continue
        
        G = to_networkx(data, to_undirected=True)
        A = nx.adjacency_matrix(G).toarray().astype(float)
        
        # TODO: Run brute force and record results
        pass
    
    return results
```

---

## Part 1: Motzkin-Straus Formulation

### The Key Question

We just saw that brute force is exponential. Can we do better?

**The Motzkin-Straus theorem provides a remarkable answer:** We can transform the discrete combinatorial problem into a continuous optimization problem!

### From Discrete to Continuous

**The Discrete Problem:**
- Find the largest subset $C \subseteq V$ where all pairs are connected
- This requires searching through $2^n$ subsets

**The Continuous Problem:**
- Find a point $\mathbf{x}$ in the simplex that maximizes a quadratic function
- The simplex $\Delta_n = \{\mathbf{x} \in \mathbb{R}^n : \sum_i x_i = 1, x_i \geq 0\}$ is a continuous, convex set
- We can use calculus-based optimization.

### The Motzkin-Straus Theorem

**Theorem:** Let $G = (V, E)$ be a graph with adjacency matrix $\mathbf{A}$. Define:

$$
f(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x} = \sum_{i,j} x_i A_{ij} x_j
$$

Then the **global maximum** of $f$ over the simplex $\Delta_n$ satisfies:

$$
\max_{\mathbf{x} \in \Delta_n} f(\mathbf{x}) = 1 - \frac{1}{\omega(G)}
$$

where $\omega(G)$ is the **clique number** (size of maximum clique).

Moreover, the maximizer $\mathbf{x}^*$ has the form:

$$
x_i^* = \begin{cases}
\frac{1}{k} & \text{if } i \in C^* \\
0 & \text{otherwise}
\end{cases}
$$

where $C^*$ is a maximum clique of size $k = \omega(G)$.

### Why Does This Work? An Intuitive Explanation

Let's understand what $f(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x}$ actually computes:

$$
f(\mathbf{x}) = \sum_{i \in V} \sum_{j \in V} x_i A_{ij} x_j = 2 \sum_{(i,j) \in E} x_i x_j
$$

**Interpretation:** We're computing a weighted sum over edges, where the weight of edge $(i,j)$ is $x_i \cdot x_j$.

**Key insight:** If we want to maximize this sum:
- We should put mass ($x_i > 0$) on nodes that are **densely connected**
- The densest possible subgraph is a **clique** (every pair connected)
- Among cliques, the **largest** one gives the highest value

### Worked Example: Triangle Graph ($K_3$)

Consider the complete graph on 3 nodes (a triangle):

$$
\mathbf{A} = \begin{bmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{bmatrix}
$$

The maximum clique is all 3 nodes, so $\omega(G) = 3$.

**Let's verify the theorem:**

The simplex point for the clique $\{0, 1, 2\}$ is $\mathbf{x}^* = (\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.

Computing $f(\mathbf{x}^*)$:

$$
f(\mathbf{x}^*) = \mathbf{x}^{*T} \mathbf{A} \mathbf{x}^* = \begin{bmatrix} \frac{1}{3} & \frac{1}{3} & \frac{1}{3} \end{bmatrix} \begin{bmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} \frac{1}{3} \\ \frac{1}{3} \\ \frac{1}{3} \end{bmatrix}
$$

First, $\mathbf{A}\mathbf{x}^* = \begin{bmatrix} \frac{2}{3} \\ \frac{2}{3} \\ \frac{2}{3} \end{bmatrix}$

Then, $f(\mathbf{x}^*) = \frac{1}{3} \cdot \frac{2}{3} + \frac{1}{3} \cdot \frac{2}{3} + \frac{1}{3} \cdot \frac{2}{3} = \frac{2}{3}$

According to the theorem: $1 - \frac{1}{\omega(G)} = 1 - \frac{1}{3} = \frac{2}{3}$ ✓

### The Problem of Spurious Solutions

**Warning:** Not every maximizer corresponds to a clique!

Consider the "cherry" graph (path of length 2):

```
    0 --- 2 --- 1
```

$$
\mathbf{A} = \begin{bmatrix} 0 & 0 & 1 \\ 0 & 0 & 1 \\ 1 & 1 & 0 \end{bmatrix}
$$

The maximum clique is any edge, say $\{0, 2\}$, with size $\omega(G) = 2$.

The point $\mathbf{x}^{*} = (\frac{1}{2}, 0, \frac{1}{2})$ gives $f(\mathbf{x}^{*}) = 0.5 = 1 - \frac{1}{2}$ ✓

**But also:** $\tilde{\mathbf{x}} = (\frac{1}{4}, \frac{1}{4}, \frac{1}{2})$ gives the same value!

$$
f(\tilde{\mathbf{x}}) = 2 \cdot (\frac{1}{4} \cdot \frac{1}{2} + \frac{1}{4} \cdot \frac{1}{2}) = 2 \cdot \frac{1}{4} = 0.5
$$

But $\{0, 1, 2\}$ is **not** a clique (0 and 1 are not connected)!

This is called a **spurious solution**.

### Regularization to Avoid Spurious Solutions

To eliminate spurious solutions, we use the **regularized objective**:

$$
\hat{f}(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x} + \frac{1}{2} \mathbf{x}^T \mathbf{x} = \mathbf{x}^T \left(\mathbf{A} + \frac{1}{2}\mathbf{I}\right) \mathbf{x}
$$

**Why does this help?**

The term $\frac{1}{2}\mathbf{x}^T\mathbf{x} = \frac{1}{2}\sum_i x_i^2$ penalizes "spread out" solutions.

For the cherry graph:
- $\hat{f}(\mathbf{x}^{*}) = 0.5 + \frac{1}{2}((\frac{1}{2})^2 + 0 + (\frac{1}{2})^2) = 0.5 + 0.25 = 0.75$
- $\hat{f}(\tilde{\mathbf{x}}) = 0.5 + \frac{1}{2}((\frac{1}{4})^2 + (\frac{1}{4})^2 + (\frac{1}{2})^2) = 0.5 + 0.1875 = 0.6875$

Now the true clique solution $\mathbf{x}^{*}$ has a **higher** value than the spurious solution!

### Implementation Task

**Task 1.1:** Implement the objective functions.

```python
def motzkin_straus_objective(x, A):
    """
    Compute the Motzkin-Straus objective f(x) = x^T A x
    
    Args:
        x: numpy array of shape (n,) - point in the simplex
        A: numpy array of shape (n, n) - adjacency matrix
    
    Returns:
        float: objective value
    """
    # TODO: Implement this function
    pass


def regularized_objective(x, A):
    """
    Compute the regularized objective f_hat(x) = x^T (A + 0.5*I) x
    
    Args:
        x: numpy array of shape (n,) - point in the simplex
        A: numpy array of shape (n, n) - adjacency matrix
    
    Returns:
        float: regularized objective value
    """
    # TODO: Implement this function
    pass
```

**Task 1.2:** Verify the Motzkin-Straus theorem on several examples.

```python
def verify_motzkin_straus():
    """
    Verify the Motzkin-Straus theorem on test graphs.
    """
    # Test 1: Triangle (K3) - maximum clique size 3
    A1 = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ], dtype=float)
    
    # TODO: Find maximum clique with brute force
    # TODO: Compute simplex point for the clique
    # TODO: Compute f(x*) and verify it equals 1 - 1/k
    
    # Test 2: Square with diagonal - maximum clique size 3
    # Nodes: 0-1-2-3 form a square, plus diagonal 0-2
    A2 = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=float)
    
    # TODO: Same verification
    
    # Test 3: Cherry graph - verify spurious solutions
    A3 = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 0]
    ], dtype=float)
    
    # TODO: Show that regularization distinguishes true from spurious solutions

verify_motzkin_straus()
```


### Key Takeaway

**The Motzkin-Straus theorem tells us WHAT to optimize, but not HOW.**

We've transformed finding a maximum clique into:
$$\text{maximize } f(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x} \text{ subject to } \mathbf{x} \in \Delta_n$$

But we still need an algorithm to find this maximum. That's where **Replicator Dynamics** comes in!

---

## Part 2: Replicator Dynamics

### The Optimization Challenge

We need to maximize $f(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x}$ over the simplex.

Standard gradient ascent won't work directly because:
1. We must stay in the simplex (constraints!)
2. The problem is non-convex (multiple local maxima)

**Replicator Dynamics** is an elegant algorithm from evolutionary game theory that naturally respects the simplex constraints. This approach was popularized in computer vision by Pelillo et al. in their seminal work on graph matching and relaxation labeling processes.

### The Intuition: Evolution by Natural Selection

Imagine a population of $n$ species, where $x_i$ represents the proportion of species $i$.

- The **fitness** of species $i$ depends on interactions with other species
- Species with above-average fitness **grow**
- Species with below-average fitness **shrink**
- This continues until an **equilibrium** is reached

The remarkable fact: **This equilibrium corresponds to the maximum clique!**

### The Mathematics

The connection between Replicator Dynamics and the Maximum Clique problem comes from evolutionary game theory. Consider the adjacency matrix $\mathbf{A}$ as a **payoff matrix** where $A_{ij}$ represents the payoff when strategy $i$ interacts with strategy $j$.

**Support (or fitness) of species $i$:**
$$\pi_i = (\mathbf{A}\mathbf{x})_i = \sum_j A_{ij} x_j$$

This represents the expected payoff for an individual playing strategy $i$ against the mixed population $\mathbf{x}$.

**Average fitness of the population:**
$$\phi = \mathbf{x}^T \mathbf{A} \mathbf{x} = \sum_i x_i \pi_i$$

### Discrete Replicator Dynamics

Following the formulation from Pelillo's work on relaxation labeling and game-theoretic methods, the discrete-time Replicator Dynamics have an elegant multiplicative form:

$$
x_i(t+1) = x_i(t) \cdot \frac{(\mathbf{A}\mathbf{x}(t))_i}{\sum_j x_j(t) \cdot (\mathbf{A}\mathbf{x}(t))_j}
$$

In matrix notation, this can be written compactly as:

$$
\mathbf{x}(t+1) = \frac{\mathbf{x}(t) \odot (\mathbf{A}\mathbf{x}(t))}{\mathbf{x}(t)^T \mathbf{A} \mathbf{x}(t)}
$$

where $\odot$ denotes element-wise (Hadamard) multiplication.

**Even more compactly**, if we denote the normalization implicitly:

$$
\mathbf{x} \leftarrow \mathbf{x} \odot (\mathbf{A}\mathbf{x}), \quad \text{then normalize so } \sum_i x_i = 1
$$


### Key Theoretical Properties

**Theorem (Pelillo et al.):** The discrete Replicator Dynamics have the following properties:

1. **Simplex Invariance:** If $\mathbf{x}(0) \in \Delta_n$, then $\mathbf{x}(t) \in \Delta_n$ for all $t \geq 0$.

2. **Monotonic Increase:** The objective function $f(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x}$ is strictly increasing along non-constant trajectories:
$$f(\mathbf{x}(t+1)) \geq f(\mathbf{x}(t))$$
with equality if and only if $\mathbf{x}(t)$ is a stationary point.

3. **Convergence:** The dynamics converge to a stationary point (Nash equilibrium).

4. **Clique Correspondence:** For the Maximum Clique problem with regularized adjacency $\mathbf{A} + \frac{1}{2}\mathbf{I}$, stationary points correspond to maximal cliques, and global maxima correspond to maximum cliques.

### Algorithm

```
Algorithm: Replicator Dynamics (Pelillo formulation)
─────────────────────────────────────────────────────
Input: Adjacency matrix A, max iterations T
Output: Converged simplex point x*

1. n ← number of nodes
2. A_reg ← A + 0.5 * I  // Regularization to avoid spurious solutions
3. x ← (1/n, 1/n, ..., 1/n)  // Start at barycenter

4. For t = 1 to T:
   a. x ← x ⊙ (A_reg @ x)     // Element-wise multiplication
   b. x ← x / sum(x)           // Normalize to stay in simplex

5. Return x
```

### Implementation Task

**Task 2.1:** Implement Replicator Dynamics following the Pelillo formulation.

```python
def replicator_dynamics(A, max_iter=1000, tol=1e-12, use_regularization=True):
    """
    Run Replicator Dynamics to find the maximum clique.
    
    This follows the formulation from Pelillo et al.:
    x_new = x * (A @ x)
    x_new = x_new / x_new.sum()
    
    Args:
        A: numpy array of shape (n, n) - adjacency matrix
        max_iter: int - maximum number of iterations
        tol: float - convergence tolerance
        use_regularization: bool - whether to use A + 0.5*I
    
    Returns:
        x: numpy array of shape (n,) - final simplex point
        history: dict with 'objectives' and 'iterations'
    """
    n = A.shape[0]
    
    # Regularization (Pelillo et al. modification to avoid spurious solutions)
    if use_regularization:
        A_reg = A + 0.5 * np.eye(n)
    else:
        A_reg = A.copy()
    
    # Initialize at barycenter
    x = np.ones(n) / n
    
    history = {
        'objectives': [x @ A_reg @ x],
        'iterations': 0
    }
    
    for t in range(max_iter):
        x_old = x.copy()
        
        # TODO: Implement the Pelillo RD update
        # Step 1: x_new = x * (A_reg @ x)  [element-wise multiplication]
        # Step 2: x_new = x_new / x_new.sum()  [normalize]
        # Step 3: Record objective, check convergence
        pass
    
    return x, history
```


**Task 2.2:** Implement the clique decoder.

Once RD converges, nodes in the maximum clique will have $x_i \approx \frac{1}{k}$ (where $k$ is the clique size), while nodes outside will have $x_i \approx 0$.

```python
def decode_clique(x, A, threshold=1e-6):
    """
    Extract the maximum clique from the RD solution.
    
    Strategy: 
    1. Sort nodes by x value (descending)
    2. Greedily add nodes that maintain clique property
    
    Args:
        x: numpy array of shape (n,) - RD solution
        A: numpy array of shape (n, n) - adjacency matrix
        threshold: float - minimum x value to consider
    
    Returns:
        clique: list of node indices
    """
    # TODO: Implement the greedy decoder
    pass
```

**Algorithm for decoding:**

```
Algorithm: Clique Decoder
─────────────────────────────────────
Input: Solution x, Adjacency A, threshold δ
Output: Clique C

1. indices ← argsort(-x)  // Sort descending by x value
2. C ← []

3. For i in indices:
   a. If x[i] < δ and |C| > 0:
      break  // Remaining nodes have negligible mass
   b. If Can_Extend(C, i, A):
      C ← C ∪ {i}

4. Return C

Function Can_Extend(C, i, A):
   For each j in C:
      If A[j, i] = 0:
         Return False
   Return True
```

**Task 2.3:** Visualize RD convergence.

```python
def visualize_rd(A, graph_name="Test Graph"):
    """
    Run RD and create visualization of convergence.
    """
    x_final, history = replicator_dynamics(A)
    n = A.shape[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Objective value f(x(t)) over time
    # Shows the monotonic increase property
    
    # Plot 2: Final x values as bar chart
    # Nodes in the clique should have high values (~1/k)
    # Nodes outside should have values near 0
    
    # Plot 3: The graph with nodes colored by x value
    
    plt.suptitle(f"Replicator Dynamics: {graph_name}")
    plt.tight_layout()
    plt.show()
    
    return x_final
```

**Task 2.4:** Apply to IMDB and COLLAB datasets.

```python
def evaluate_rd_on_dataset(dataset, dataset_name, num_graphs=50):
    """
    Evaluate Replicator Dynamics on a dataset.
    """
    results = []
    
    for i in range(min(num_graphs, len(dataset))):
        data = dataset[i]
        G = to_networkx(data, to_undirected=True)
        A = nx.adjacency_matrix(G).toarray().astype(float)
        
        start_time = time.time()
        x_final, history = replicator_dynamics(A, max_iter=1000)
        clique = decode_clique(x_final, A)
        elapsed = time.time() - start_time
        
        # Verify it's a valid clique
        valid = is_clique(clique, A)
        
        results.append({
            'graph_idx': i,
            'num_nodes': data.num_nodes,
            'clique_size': len(clique),
            'clique': clique,
            'valid': valid,
            'time': elapsed,
            'iterations': history['iterations']
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{num_graphs} graphs")
    
    # Summary statistics
    sizes = [r['clique_size'] for r in results]
    times = [r['time'] for r in results]
    valid_count = sum(r['valid'] for r in results)
    
    print(f"\n{dataset_name} Results:")
    print(f"  Mean clique size: {np.mean(sizes):.2f} ± {np.std(sizes):.2f}")
    print(f"  Mean time: {np.mean(times):.4f} seconds")
    print(f"  Valid cliques: {valid_count}/{len(results)}")
    
    return results
```


---

## Part 3: GNN + Replicator Dynamics

### Motivation: The Initialization Problem

Replicator Dynamics always starts from the **barycenter** (uniform distribution). This is a "neutral" starting point that doesn't use any information about the graph structure.

**Question:** Can we learn a better starting point?

If we could initialize RD **closer to the optimal solution**, it would:
1. Converge faster (fewer iterations)
2. Be less likely to get stuck in local optima

### The Hybrid Approach

We train a GNN to predict which nodes are likely to be in the maximum clique. The GNN output (after softmax) gives us a point in the simplex, which we use to initialize RD.

**Training Pipeline:**
```
Node Features → GNN → Linear → Softmax → p → RD (2-3 iter) → p'
                                                    ↓
                                        Loss = -p'^T A p'
```

**Inference Pipeline:**
```
Node Features → GNN → Linear → Softmax → p → RD (100 iter) → p* → Decoder → Clique
```

### Why Include RD in Training?

If we only trained the GNN with loss $-\mathbf{p}^T \mathbf{A} \mathbf{p}$, it might learn solutions that are hard for RD to refine.

By including a few RD iterations in the training loop:
1. The GNN learns to produce initializations that RD can improve
2. The gradients flow through RD, teaching the GNN what makes a "good" starting point

### Implementation Task

**Task 3.1:** Compute node features.

```python
def compute_node_features(data):
    """
    Compute informative node features for the GNN.
    
    Features:
    - Normalized degree
    - Clustering coefficient
    - Log(degree + 1)
    
    Args:
        data: PyTorch Geometric Data object
    
    Returns:
        x: torch.Tensor of shape (num_nodes, num_features)
    """
    G = to_networkx(data, to_undirected=True)
    n = G.number_of_nodes()
    
    features = []
    
    # TODO: Compute features for each node
    # Degree
    # Clustering coefficient (nx.clustering)
    # You can add more features
    
    return torch.tensor(features, dtype=torch.float)
```

**Task 3.2:** Implement the GNN model.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MaxCliqueGNN(nn.Module):
    """
    GNN for predicting initial simplex point for Maximum Clique.
    """
    
    def __init__(self, input_dim, hidden_dim=32, num_layers=2):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)
        
        Returns:
            p: Simplex point (num_nodes,)
        """
        # TODO: Implement forward pass
        # 1. Apply GCN layers with ReLU activation
        # 2. Apply output linear layer
        # 3. Apply softmax to get valid simplex point
        pass
```

**Task 3.3:** Implement differentiable Replicator Dynamics.

```python
def differentiable_rd_step(p, A):
    """
    One step of differentiable Replicator Dynamics.
    
    Args:
        p: torch.Tensor (n,) - current simplex point
        A: torch.Tensor (n, n) - adjacency matrix (with regularization)
    
    Returns:
        p_new: torch.Tensor (n,) - updated simplex point
    """
    # TODO: Implement one RD step in PyTorch
    # This must be differentiable!
    pass


def differentiable_rd(p, A, num_iterations=3):
    """
    Multiple steps of differentiable RD.
    """
    A_reg = A + 0.5 * torch.eye(A.shape[0], device=A.device)
    
    for _ in range(num_iterations):
        p = differentiable_rd_step(p, A_reg)
    
    return p
```

**Task 3.4:** Implement the loss function.

```python
def maxclique_loss(p, A):
    """
    Motzkin-Straus loss (negative because we minimize).
    
    Args:
        p: torch.Tensor (n,) - simplex point
        A: torch.Tensor (n, n) - adjacency matrix
    
    Returns:
        loss: scalar tensor
    """
    # TODO: Return -p^T A p
    pass
```

**Task 3.5:** Implement the training loop.

```python
def train_gnn(model, dataset, num_epochs=100, lr=0.01, rd_train_iters=3):
    """
    Train the GNN model.
    
    Args:
        model: MaxCliqueGNN
        dataset: PyTorch Geometric dataset
        num_epochs: number of training epochs
        lr: learning rate
        rd_train_iters: RD iterations during training
    
    Returns:
        model: trained model
        losses: list of epoch losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for data in dataset:
            optimizer.zero_grad()
            
            # TODO: Implement training step
            # 1. Compute node features
            # 2. Forward pass through GNN to get p
            # 3. Apply differentiable RD to get p'
            # 4. Compute loss
            # 5. Backward and optimize
            
            pass
        
        epoch_loss /= len(dataset)
        losses.append(epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    return model, losses
```

**Task 3.6:** Implement evaluation.

```python
def evaluate_gnn(model, dataset, rd_test_iters=100):
    """
    Evaluate the trained model.
    
    Args:
        model: trained MaxCliqueGNN
        dataset: PyTorch Geometric dataset
        rd_test_iters: RD iterations during inference
    
    Returns:
        results: list of evaluation results
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for i, data in enumerate(dataset):
            # TODO: Implement evaluation
            # 1. Compute features
            # 2. GNN forward pass
            # 3. Many RD iterations
            # 4. Decode clique
            # 5. Verify and record
            pass
    
    return results
```

---

## Exercises

### Exercise 1: Brute Force Baseline (Part 0)
1. Implement `is_clique()` and `brute_force_max_clique()`
2. Test on small synthetic graphs (4-8 nodes)
3. Apply to first 20 graphs of IMDB (skip graphs with >25 nodes)
4. Report: clique sizes found and computation times

### Exercise 2: Motzkin-Straus Verification (Part 1)
1. Implement both objective functions
2. Create 3 test graphs with known maximum cliques
3. Verify the theorem holds for each
4. Demonstrate the spurious solution problem on the cherry graph
5. Show that regularization fixes it

### Exercise 3: Replicator Dynamics (Part 2)
1. Implement RD with the decoder
2. Visualize convergence on a small example
3. Apply to IMDB and COLLAB (50 graphs each)
4. Compare clique sizes with brute force (where available)
5. Report: accuracy, mean clique size, computation time

### Exercise 4: GNN + RD Hybrid (Part 3)
1. Implement the complete pipeline
2. Train on IMDB, evaluate on held-out graphs
3. Compare three approaches:
   - Brute force (ground truth, where feasible)
   - Pure RD from barycenter
   - GNN + RD
4. Ablation: vary number of RD iterations (train: 0,1,3,5; test: 10,50,100)

### Exercise 5: Final Comparison
Create a summary table:

| Method | IMDB Clique Size | IMDB Time | COLLAB Clique Size | COLLAB Time |
|--------|------------------|-----------|--------------------| ------------|
| Brute Force | - | - | - | - |
| Replicator Dynamics | - | - | - | - |
| GNN + RD | - | - | - | - |

---

## Deliverables

1. **Code**: Complete, documented implementations
2. **Results**: Tables and visualizations
3. **Report** (3-4 paragraphs):
   - Explain why continuous relaxation works
   - Analyze when GNN+RD outperforms pure RD
   - Discuss limitations and potential improvements

---

## References

1. Motzkin, T.S., & Straus, E.G. (1965). Maxima for graphs and a new proof of a theorem of Turán.
2. Pelillo, M. (1999). Replicator Equations, Maximal Cliques, and Graph Isomorphism. Neural Computation.
3. Bomze, I.M. (1997). Evolution towards the Maximum Clique. Journal of Global Optimization.
4. Min, Y., Wenkel, F., et al. (2022). Can Hybrid Geometric Scattering Networks Help Solve the Maximum Clique Problem? NeurIPS.
3. Elezi, I., Vascon, S., Torcinovich, A., Pelillo, M., Leal-Taixé, L. (2020). *The Group Loss for Deep Metric Learning*. ECCV. (Uses replicator dynamics for label propagation)