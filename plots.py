"""
Shared plotting utilities for graph tutorial notebooks.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Check for optional dependencies
try:
    from ipysigma import Sigma
    IPYSIGMA_AVAILABLE = True
except ImportError:
    IPYSIGMA_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def visualize_graph(G, node_color_attr=None, node_size_attr=None,
                    edge_color_attr=None, default_node_size=5,
                    height=800, hide_null_colors=True,
                    **sigma_kwargs):
    """
    Visualize a NetworkX DiGraph using ipysigma.

    Parameters
    ----------
    G : nx.DiGraph
        The graph to visualize
    node_color_attr : str, optional
        Node attribute name to use for coloring (e.g., 'community', 'degree')
    node_size_attr : str, optional
        Node attribute name to use for sizing (e.g., 'degree', 'centrality')
        If None, uses default_node_size for all nodes
    edge_color_attr : str, optional
        Edge attribute name to use for coloring (e.g., 'weight')
    default_node_size : int
        Default node size when node_size_attr is None
    height : int
        Height of the widget in pixels
    hide_null_colors : bool
        If True, nodes with None/null color attribute values are grouped as '__unlabeled__'
    **sigma_kwargs : dict
        Additional arguments passed to Sigma()

    Returns
    -------
    Sigma
        The ipysigma widget

    Raises
    ------
    ImportError
        If ipysigma is not installed
    """
    if not IPYSIGMA_AVAILABLE:
        raise ImportError("ipysigma is required for visualize_graph(). Install with: pip install ipysigma")

    # Build kwargs dynamically - only include if specified
    kwargs = {'height': height, **sigma_kwargs}

    if node_color_attr:
        if hide_null_colors:
            # Create a display attribute that keeps real categories but assigns
            # unique random IDs to unlabeled nodes. This way:
            # - Real categories get colored by ipysigma (top 10 by frequency)
            # - Unlabeled nodes each become a "rare" category (count=1) and
            #   fall outside top 10, getting the default gray color
            display_attr = f'_display_{node_color_attr}'
            attrs = nx.get_node_attributes(G, node_color_attr)

            unlabeled_counter = 0
            for node in G.nodes():
                val = attrs.get(node)
                # Treat None and legacy '__unlabeled__' as unlabeled
                if val is None or val == '__unlabeled__':
                    # Each unlabeled node gets a unique "rare" category
                    G.nodes[node][display_attr] = f'_unlabeled_{unlabeled_counter}'
                    unlabeled_counter += 1
                else:
                    G.nodes[node][display_attr] = val

            # Use the display attribute for coloring, original stays intact
            kwargs['node_color'] = display_attr
        else:
            kwargs['node_color'] = node_color_attr

    if node_size_attr:
        kwargs['node_size'] = node_size_attr
    else:
        # Pass a mapping: all nodes get the same size
        kwargs['node_size'] = {n: default_node_size for n in G.nodes()}

    if edge_color_attr:
        kwargs['edge_color'] = edge_color_attr

    return Sigma(G, **kwargs)


def plot_adjacency_matrix(G, sort_by='degree', community_attr=None,
                          node_attr=None, size=800, marker_size=1,
                          show_boundaries=True, boundary_positions=None):
    """
    Plot adjacency matrix of a NetworkX DiGraph using Plotly.

    Parameters
    ----------
    G : nx.DiGraph
        The graph to visualize
    sort_by : str
        How to order nodes: 'degree', 'in_degree', 'out_degree', 'community', 'attribute', or None
    community_attr : str, optional
        Node attribute name containing community labels (used when sort_by='community')
    node_attr : str, optional
        Node attribute name to sort by (used when sort_by='attribute')
    size : int
        Figure width and height in pixels
    marker_size : int
        Size of each dot in the matrix
    show_boundaries : bool
        Whether to show community boundaries when sorted by community
    boundary_positions : list, optional
        Manual boundary positions (list of indices where communities end)

    Returns
    -------
    go.Figure
        Plotly figure object

    Raises
    ------
    ImportError
        If plotly is not installed
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for plot_adjacency_matrix(). Install with: pip install plotly")

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    A = nx.adjacency_matrix(G, nodelist=nodes)

    # Get nonzero coordinates
    rows, cols = A.nonzero()

    # Determine sort order
    if sort_by == 'community' and community_attr:
        labels = [G.nodes[n].get(community_attr, 0) for n in nodes]
        order = np.argsort(labels)
    elif sort_by == 'attribute' and node_attr:
        values = [G.nodes[n].get(node_attr, 0) for n in nodes]
        order = np.argsort(values)[::-1]
    elif sort_by == 'in_degree':
        values = [G.in_degree(n) for n in nodes]
        order = np.argsort(values)[::-1]
    elif sort_by == 'out_degree':
        values = [G.out_degree(n) for n in nodes]
        order = np.argsort(values)[::-1]
    elif sort_by == 'degree':
        values = [G.degree(n) for n in nodes]
        order = np.argsort(values)[::-1]
    else:
        order = np.arange(len(nodes))

    # Apply sorting
    row_map = {old: new for new, old in enumerate(order)}
    rows_sorted = [row_map[r] for r in rows]
    cols_sorted = [row_map[c] for c in cols]
    sorted_nodes = [nodes[i] for i in order]

    # Create figure
    fig = go.Figure(data=go.Scattergl(
        x=cols_sorted,
        y=rows_sorted,
        mode='markers',
        marker=dict(size=marker_size, color='black'),
        hovertemplate='%{customdata[0]} → %{customdata[1]}<extra></extra>',
        customdata=[(sorted_nodes[r], sorted_nodes[c]) for r, c in zip(rows_sorted, cols_sorted)]
    ))

    # Add community boundaries if requested
    shapes = []
    if show_boundaries and boundary_positions:
        for pos in boundary_positions:
            # Horizontal line
            shapes.append(dict(
                type='line', x0=-0.5, x1=len(nodes)-0.5, y0=pos-0.5, y1=pos-0.5,
                line=dict(color='red', width=1)
            ))
            # Vertical line
            shapes.append(dict(
                type='line', x0=pos-0.5, x1=pos-0.5, y0=-0.5, y1=len(nodes)-0.5,
                line=dict(color='red', width=1)
            ))

    fig.update_layout(
        width=size, height=size,
        title=f'Adjacency Matrix ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges, sorted by {sort_by})',
        yaxis=dict(autorange='reversed', scaleanchor='x', title='Pre-synaptic neuron'),
        xaxis=dict(constrain='domain', title='Post-synaptic neuron'),
        shapes=shapes
    )

    return fig


def draw_motif_graph(ax, motif_nx, label=None, fontsize=7.0):
    """
    Draw a small 3-node motif graph in the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on (typically an inset)
    motif_nx : nx.DiGraph
        NetworkX graph representing the motif (3 nodes)
    label : str, optional
        Label to show below the motif (e.g., 'M0')
    fontsize : float, optional
        Font size for the label. Default 7.
    """
    # Use circular layout scaled down to fit in small inset
    pos = nx.circular_layout(motif_nx, scale=0.35)
    nx.draw(motif_nx, pos, ax=ax,
            node_color='lightgray', node_size=60,
            edge_color='black', arrows=True, arrowsize=5,
            arrowstyle='->', width=1.0, with_labels=False)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.axis('off')
    if label:
        ax.set_title(label, fontsize=fontsize, pad=1)


def nx_to_gt(G_nx):
    """
    Convert a NetworkX graph to graph-tool Graph.

    Parameters
    ----------
    G_nx : nx.Graph or nx.DiGraph
        NetworkX graph

    Returns
    -------
    G_gt : graph_tool.Graph
        Graph-tool equivalent
    node_map : dict
        Mapping from NetworkX node labels to graph-tool vertex indices
    """
    import graph_tool.all as gt

    G_gt = gt.Graph(directed=G_nx.is_directed())
    node_map = {n: i for i, n in enumerate(G_nx.nodes())}
    G_gt.add_vertex(len(node_map))
    for u, v in G_nx.edges():
        G_gt.add_edge(node_map[u], node_map[v])
    return G_gt, node_map


def gt_to_nx(G_gt):
    """
    Convert a graph-tool Graph to NetworkX graph.

    Parameters
    ----------
    G_gt : graph_tool.Graph
        Graph-tool graph

    Returns
    -------
    nx.DiGraph or nx.Graph
        NetworkX equivalent
    """
    if G_gt.is_directed():
        G_nx = nx.DiGraph()
    else:
        G_nx = nx.Graph()
    G_nx.add_nodes_from(range(G_gt.num_vertices()))
    for e in G_gt.edges():
        G_nx.add_edge(int(e.source()), int(e.target()))
    return G_nx


def gt_motif_to_nx(motif_gt):
    """
    Convert a graph-tool motif Graph to NetworkX DiGraph.

    Alias for gt_to_nx() for backwards compatibility.

    Parameters
    ----------
    motif_gt : graph_tool.Graph
        Graph-tool graph representing a 3-node motif

    Returns
    -------
    nx.DiGraph
        NetworkX equivalent
    """
    return gt_to_nx(motif_gt)


def get_canonical_triads():
    """
    Generate all 16 canonical 3-node directed motif graphs.

    Returns list of nx.DiGraph in canonical order (M0-M15), matching
    the order returned by graph-tool's gt.motifs(G, k=3).

    The 16 triads correspond to the MAN classification:
    M0:  003 (empty)
    M1:  012 (single edge)
    M2:  102 (mutual edge)
    M3:  021D (out-star)
    M4:  021U (in-star)
    M5:  021C (chain)
    M6:  111D (out-star + mutual)
    M7:  111U (in-star + mutual)
    M8:  030T (transitive)
    M9:  030C (cycle)
    M10: 201 (mutual + single out)
    M11: 120D (out-star + mutual pair)
    M12: 120U (in-star + mutual pair)
    M13: 120C (chain + mutual)
    M14: 210 (near-complete)
    M15: 300 (complete/clique)
    """
    # Edge lists for each canonical triad (nodes are 0, 1, 2)
    triad_edges = [
        [],                          # M0:  003 - empty
        [(0, 1)],                    # M1:  012 - single edge
        [(0, 1), (1, 0)],            # M2:  102 - mutual
        [(0, 1), (0, 2)],            # M3:  021D - out-star
        [(1, 0), (2, 0)],            # M4:  021U - in-star
        [(0, 1), (2, 0)],            # M5:  021C - chain
        [(0, 1), (1, 0), (0, 2)],    # M6:  111D - mutual + out
        [(0, 1), (1, 0), (2, 0)],    # M7:  111U - mutual + in
        [(0, 1), (0, 2), (1, 2)],    # M8:  030T - transitive
        [(0, 1), (1, 2), (2, 0)],    # M9:  030C - cycle
        [(0, 1), (1, 0), (2, 0), (2, 1)],  # M10: 201
        [(0, 1), (1, 0), (0, 2), (1, 2)],  # M11: 120D
        [(0, 1), (1, 0), (2, 0), (2, 1)],  # M12: 120U
        [(0, 1), (1, 0), (0, 2), (2, 1)],  # M13: 120C
        [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2)],  # M14: 210
        [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)],  # M15: 300 - clique
    ]

    triads = []
    for edges in triad_edges:
        g = nx.DiGraph()
        g.add_nodes_from([0, 1, 2])
        g.add_edges_from(edges)
        triads.append(g)

    return triads


def align_triad_counts(motif_data):
    """
    Align motif counts from multiple gt.motifs() calls to 16 triad classes.

    Parameters
    ----------
    motif_data : list of tuples
        List of (motifs, counts) tuples from gt.motifs(G, k=3)

    Returns
    -------
    counts : np.ndarray
        Shape (n_graphs, 16) array of motif counts
    motif_graphs : list
        List of 16 nx.DiGraph representing each triad class (canonical)
    """
    all_counts = []

    for motifs, counts in motif_data:
        # Pad counts to 16 elements
        padded = np.zeros(16)
        padded[:len(counts)] = counts
        all_counts.append(padded)

    # Always return canonical triads (not dependent on observed data)
    return np.array(all_counts), get_canonical_triads()


def plot_motif_significance(results_dict, motif_graphs, observed_counts=None,
                            null_models=None, axes=None, title=None,
                            figsize=None, save_path=None, show_legend=True):
    """
    Plot motif significance z-scores with mini motif diagrams on x-axis.

    Parameters
    ----------
    results_dict : dict
        Dictionary with keys as null model names, values containing:
        - 'z_scores': array of z-scores per motif
        - 'p_values': array of p-values per motif
        Or for legacy format (from gt.motif_significance):
        - {motif_idx: {'z_score': float, 'p_value': float, 'observed': int}}
    motif_graphs : list
        List of nx.DiGraph objects representing each motif topology
        (indices correspond to motif indices)
    observed_counts : array-like, optional
        Array of observed counts per motif. Used to filter motifs with count > 0.
        If None, uses all motifs from results_dict.
    null_models : list, optional
        List of null model names to plot. If None, uses all keys in results_dict
    axes : list of matplotlib.axes.Axes, optional
        List of axes to plot on (one per null model). If None, creates new figure.
        Use this to embed plots in a larger figure (e.g., rows=networks, cols=methods).
    title : str, optional
        Figure title (only used when axes is None)
    figsize : tuple, optional
        Figure size (only used when axes is None). If None, auto-calculated.
    save_path : str, optional
        Path to save the figure (only used when axes is None)
    show_legend : bool, optional
        Whether to show the legend at bottom. Default True.
        Set to False when embedding in larger figures.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if axes is None, otherwise None

    Examples
    --------
    # Standalone plot
    fig = plot_motif_significance(results, motif_graphs)

    # Embedded in larger figure (rows=networks, cols=methods)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plot_motif_significance(results_A, motifs, axes=axes[0, :], show_legend=False)
    plot_motif_significance(results_B, motifs, axes=axes[1, :], show_legend=False)
    """
    # Determine which null models to plot
    if null_models is None:
        null_models = list(results_dict.keys())

    # Model display names
    model_names = {
        'erdos': 'Erdos-Renyi',
        'configuration': 'Configuration',
        'constrained-configuration': 'Constrained Configuration'
    }

    n_models = len(null_models)

    # Detect data format and extract motif indices
    first_model = null_models[0]
    first_data = results_dict[first_model]

    if isinstance(first_data, dict) and 'z_scores' in first_data:
        # New format: {'z_scores': array, 'p_values': array}
        data_format = 'new'
        if observed_counts is not None:
            valid_motifs = np.where(np.array(observed_counts) > 0)[0]
        else:
            valid_motifs = np.arange(len(first_data['z_scores']))
    else:
        # Legacy format: {motif_idx: {'z_score': float, ...}}
        data_format = 'legacy'
        valid_motifs = sorted(first_data.keys())

    n_motifs = len(valid_motifs)

    # Handle axes: create figure if not provided
    created_figure = axes is None
    if created_figure:
        if figsize is None:
            figsize = (max(12, n_motifs * 1.2), 4 * n_models + 1)
        fig, axes = plt.subplots(n_models, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()
    else:
        fig = None
        # Ensure axes is a list/array
        if not hasattr(axes, '__len__'):
            axes = [axes]

    if len(axes) < n_models:
        raise ValueError(f"Need {n_models} axes for {n_models} null models, got {len(axes)}")

    for row_idx, model in enumerate(null_models):
        ax = axes[row_idx]
        data = results_dict[model]

        # Extract z-scores and p-values based on format
        if data_format == 'new':
            z_scores = [data['z_scores'][m] for m in valid_motifs]
            p_values = [data['p_values'][m] for m in valid_motifs]
        else:
            z_scores = [data[m]['z_score'] for m in valid_motifs]
            p_values = [data[m]['p_value'] for m in valid_motifs]

        # Color by direction
        colors = ['green' if z > 0 else 'red' for z in z_scores]

        # Create bar plot
        x_pos = np.arange(n_motifs)
        ax.bar(x_pos, z_scores, color=colors, edgecolor='black', alpha=0.7)

        # Add significance threshold lines
        ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=-2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=0, color='black', linewidth=1)

        # Add significance stars (positioned just above/below bar tops)
        for i, (z, p) in enumerate(zip(z_scores, p_values)):
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            else:
                stars = ''
            if stars:
                # Use 2% of bar height as offset, with minimum of 0.02
                offset = max(abs(z) * 0.02, 0.02)
                y_pos = z + (offset if z > 0 else -offset)
                va = 'bottom' if z > 0 else 'top'
                ax.text(i, y_pos, stars, ha='center', va=va, fontsize=10, fontweight='bold', color='black')

        # Formatting
        ax.set_ylabel('Z-score', fontsize=11)
        display_name = model_names.get(model, model.replace('-', ' ').title())
        ax.set_title(display_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks([])  # Remove x-ticks, will add motif graphs instead
        ax.set_xlim(-0.5, n_motifs - 0.5)

        # Draw mini motif graphs on x-axis
        for bar_idx, motif_idx in enumerate(valid_motifs):
            # Create inset axes for mini graph
            inset = ax.inset_axes(
                [bar_idx / n_motifs + 0.01, -0.22, 0.9 / n_motifs, 0.18],
                transform=ax.transAxes
            )

            # Get motif graph (handle both list and dict indexing)
            if isinstance(motif_graphs, list) and motif_idx < len(motif_graphs):
                motif_nx = motif_graphs[motif_idx]
            elif isinstance(motif_graphs, dict) and motif_idx in motif_graphs:
                motif_nx = motif_graphs[motif_idx]
            else:
                # Create empty placeholder
                motif_nx = nx.DiGraph()
                motif_nx.add_nodes_from([0, 1, 2])

            draw_motif_graph(inset, motif_nx, label=f'M{motif_idx}')

    # Only add legend/title and save if we created the figure
    if created_figure:
        if show_legend:
            fig.text(0.5, 0.02,
                     'Green = Over-represented | Red = Under-represented | '
                     '* p<0.05  ** p<0.01  *** p<0.001',
                     ha='center', fontsize=10, style='italic')

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08, hspace=0.4)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Saved: {save_path}')

        return fig

    return None


def plot_motif_counts(counts_dict, motif_labels=None, motif_graphs=None,
                      normalize=False, title=None, figsize=None, fontsize=None,
                      save_path=None):
    """
    Plot raw motif counts (or frequencies) as grouped bar chart.

    Supports multiple networks/samples for comparison, with optional
    motif diagram annotations on x-axis. Always plots in canonical order.

    Parameters
    ----------
    counts_dict : dict
        Dictionary mapping labels to count arrays. Values should be:
        - np.array with shape (n_motifs,): single network counts
        Each entry becomes one bar per motif position.
    motif_labels : list, optional
        Labels for each motif position (e.g., ['M0', 'M1', ...]).
        If None, uses indices.
    motif_graphs : list, optional
        List of nx.DiGraph for x-axis motif diagrams. If None, uses text labels only.
    normalize : bool, default=False
        If True, show frequencies (%) instead of raw counts
    title : str, optional
        Figure title
    figsize : tuple, optional
        Figure size. If None, auto-calculated based on number of motifs.
    fontsize : int or dict, optional
        Font size control. Can be:
        - int: base font size (title=base+3, labels=base, legend=base-1, motif_labels=base-4)
        - dict: {'title': int, 'labels': int, 'legend': int, 'motif_labels': int}
        If None, uses defaults (title=14, labels=11, legend=10, motif_labels=7)
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    matplotlib.figure.Figure
        The figure object

    Examples
    --------
    # Plot 5 random graph samples as separate bars
    counts, motif_graphs = align_triad_counts(motif_data)
    counts_dict = {f'Sample {i+1}': counts[i] for i in range(len(counts))}
    fig = plot_motif_counts(counts_dict, motif_graphs=motif_graphs)

    # Compare two networks
    fig = plot_motif_counts({'Network A': counts_A, 'Network B': counts_B})

    # Custom font sizes
    fig = plot_motif_counts(counts_dict, fontsize=12)  # base size
    fig = plot_motif_counts(counts_dict, fontsize={'title': 16, 'labels': 12})
    """
    # Handle fontsize parameter
    if fontsize is None:
        fs = {'title': 14, 'labels': 11, 'legend': 10, 'motif_labels': 7}
    elif isinstance(fontsize, (int, float)):
        base = fontsize
        fs = {'title': base + 3, 'labels': base, 'legend': base - 1, 'motif_labels': base - 4}
    else:
        # dict - use defaults for missing keys
        fs = {'title': 14, 'labels': 11, 'legend': 10, 'motif_labels': 7}
        fs.update(fontsize)

    # Process input - each value should be a 1D array
    processed = {}
    for label, data in counts_dict.items():
        processed[label] = np.array(data)

    # Get dimensions
    n_motifs = len(next(iter(processed.values())))
    n_networks = len(processed)

    # Use motif labels or generate them
    if motif_labels is None:
        motif_labels = [f'M{i}' for i in range(n_motifs)]

    # Normalize if requested
    if normalize:
        for label in processed:
            total = processed[label].sum()
            if total > 0:
                processed[label] = processed[label] / total * 100

    # Create figure
    if figsize is None:
        figsize = (max(12, n_motifs * 0.8), 6)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Bar positions
    x = np.arange(n_motifs)
    width = 0.8 / n_networks
    colors = sns.color_palette("tab10", n_colors=max(n_networks, 2))

    # Plot bars for each network/sample
    for i, (label, counts) in enumerate(processed.items()):
        offset = (i - n_networks / 2 + 0.5) * width
        ax.bar(x + offset, counts, width, label=label,
               color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

    # Labels and formatting
    ax.set_ylabel('Frequency (%)' if normalize else 'Count', fontsize=fs['labels'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=fs['legend'])

    # X-axis: motif diagrams or text labels
    if motif_graphs is not None and len(motif_graphs) > 0:
        ax.set_xticks([])
        ax.set_xlim(-0.5, n_motifs - 0.5)

        # Scale inset position based on motif_labels fontsize
        # Default fontsize 7 -> y_offset -0.18, larger fonts need more negative offset
        base_y_offset = -0.18
        font_scale = (fs['motif_labels'] - 7) * 0.012  # adjust offset per font size unit
        y_offset = base_y_offset - font_scale

        # Draw mini motif graphs for ALL positions
        for i in range(n_motifs):
            inset = ax.inset_axes(
                (i / n_motifs + 0.01, y_offset, 0.9 / n_motifs, 0.15),
                transform=ax.transAxes
            )
            if i < len(motif_graphs) and motif_graphs[i] is not None:
                draw_motif_graph(inset, motif_graphs[i], label=motif_labels[i],
                                fontsize=fs['motif_labels'])
            else:
                inset.axis('off')
                inset.set_title(motif_labels[i], fontsize=fs['motif_labels'])

        # Scale labelpad based on fontsize
        base_labelpad = 55
        labelpad = base_labelpad + (fs['motif_labels'] - 7) * 3
        ax.set_xlabel('Motif Type', fontsize=fs['labels'], labelpad=labelpad)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(motif_labels, rotation=45, ha='right', fontsize=fs['motif_labels'])
        ax.set_xlabel('Motif Type', fontsize=fs['labels'])

    if title:
        ax.set_title(title, fontsize=fs['title'], fontweight='bold')

    plt.tight_layout()
    if motif_graphs is not None:
        # Scale bottom margin based on fontsize
        base_bottom = 0.18
        bottom = base_bottom + (fs['motif_labels'] - 7) * 0.01
        plt.subplots_adjust(bottom=bottom)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')

    return fig


def plot_communities_comparison(G, communities_dict, pos=None, figsize=(20, 8),
                                ground_truth_key='Ground Truth', save_path=None):
    """
    Compare multiple community detection results side by side.

    Parameters
    ----------
    G : nx.DiGraph
        The graph
    communities_dict : dict
        Dictionary mapping method names to (communities, modularity) tuples
        where communities is a list of sets
    pos : dict, optional
        Node positions for layout. If None, uses spring layout
    figsize : tuple
        Figure size (width, height)
    ground_truth_key : str
        Key in communities_dict for ground truth (used for color matching)
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    from scipy.optimize import linear_sum_assignment

    n_methods = len(communities_dict)
    fig, axes = plt.subplots(2, n_methods, figsize=figsize)

    if n_methods == 1:
        axes = axes.reshape(2, 1)

    # Get positions if not provided
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    # Color palette
    color_palette = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan',
                     'olive', 'teal', 'navy', 'maroon', 'lime', 'aqua', 'fuchsia', 'silver']

    # Get ground truth for color matching
    ground_truth_communities = communities_dict.get(ground_truth_key, (None, None))[0]

    def match_clusters_to_ground_truth(detected_communities, gt_communities):
        """Match detected communities to ground truth based on maximum overlap."""
        if gt_communities is None:
            return {i: i for i in range(len(detected_communities))}

        n_detected = len(detected_communities)
        n_truth = len(gt_communities)

        # Create overlap matrix
        overlap = np.zeros((n_detected, n_truth))
        for i, det_comm in enumerate(detected_communities):
            for j, gt_comm in enumerate(gt_communities):
                overlap[i, j] = len(det_comm & gt_comm)

        # Use Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-overlap)

        # Create mapping
        mapping = {}
        for det_id, gt_id in zip(row_ind, col_ind):
            mapping[det_id] = gt_id

        # Handle unmatched
        used_gt_ids = set(col_ind)
        unused_ids = [i for i in range(max(n_detected, n_truth)) if i not in used_gt_ids]
        for det_id in range(n_detected):
            if det_id not in mapping and unused_ids:
                mapping[det_id] = unused_ids.pop(0)
            elif det_id not in mapping:
                mapping[det_id] = det_id

        return mapping

    for idx, (method_name, (communities, modularity)) in enumerate(communities_dict.items()):
        ax_graph = axes[0, idx]
        ax_matrix = axes[1, idx]

        # Match colors to ground truth
        if method_name != ground_truth_key and ground_truth_communities is not None:
            cluster_mapping = match_clusters_to_ground_truth(communities, ground_truth_communities)
        else:
            cluster_mapping = {i: i for i in range(len(communities))}

        # Assign colors to nodes
        node_to_comm = {}
        for comm_id, comm in enumerate(communities):
            mapped_id = cluster_mapping.get(comm_id, comm_id)
            for node in comm:
                node_to_comm[node] = mapped_id

        node_colors = [color_palette[node_to_comm.get(n, 0) % len(color_palette)] for n in G.nodes()]

        # Draw graph
        nx.draw(G, pos, ax=ax_graph,
                node_color=node_colors,
                node_size=200,
                with_labels=True,
                font_size=7,
                font_weight='bold',
                font_color='white',
                edge_color='gray',
                alpha=0.6,
                arrows=True,
                arrowsize=8,
                width=1.0)

        ax_graph.set_title(f'{method_name}\n{len(communities)} communities, Q = {modularity:.3f}',
                           fontsize=11, fontweight='bold')

        # Draw matrix reordered by communities
        sorted_communities = sorted(enumerate(communities),
                                    key=lambda x: cluster_mapping.get(x[0], x[0]))
        ordered_nodes = []
        for _, comm in sorted_communities:
            ordered_nodes.extend(sorted(list(comm)))

        # Get adjacency matrix
        A = nx.adjacency_matrix(G, nodelist=list(G.nodes())).toarray()
        node_list = list(G.nodes())
        node_idx = {n: i for i, n in enumerate(node_list)}

        # Reorder
        order = [node_idx[n] for n in ordered_nodes]
        A_reordered = A[np.ix_(order, order)]

        # Plot matrix
        ax_matrix.imshow(A_reordered, cmap='Blues', aspect='equal')

        # Add boundaries
        cumsum = 0
        for _, comm in sorted_communities[:-1]:
            cumsum += len(comm)
            ax_matrix.axhline(y=cumsum-0.5, color='red', linewidth=2)
            ax_matrix.axvline(x=cumsum-0.5, color='red', linewidth=2)

        ax_matrix.set_title(f'{method_name}', fontsize=11, fontweight='bold')

        if idx == 0:
            ax_matrix.set_ylabel('Pre-synaptic', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✓ Saved: {save_path}')

    return fig
