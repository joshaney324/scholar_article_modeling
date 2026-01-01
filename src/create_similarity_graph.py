import json
import networkx as nx
from pathlib import Path
import numpy as np
from pyvis.network import Network


def load_papers(data_path, num_papers=10000):

    papers = []

    print(f"Loading first {num_papers} papers from dataset...")
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_papers:
                break

            paper = json.loads(line)
            papers.append({
                'id': paper['id'],
                'title': paper['title'],
                'authors': paper['authors'],
                'authors_parsed': paper.get('authors_parsed', []),
                'categories': paper.get('categories', ''),
                'abstract': paper.get('abstract', '')
            })

            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1} papers...")

    print(f"Successfully loaded {len(papers)} papers\n")
    return papers


def parse_authors(paper):
    # Use authors_parsed if available for better name handling
    if paper['authors_parsed']:
        # Create normalized names: "lastname, firstname"
        authors = set()
        for author_parts in paper['authors_parsed']:
            if len(author_parts) >= 2:
                lastname = author_parts[0].strip().lower()
                firstname = author_parts[1].strip().lower()
                if lastname and firstname:
                    authors.add(f"{lastname}, {firstname}")
        return authors
    else:
        # Fallback: split the authors string
        authors_str = paper['authors']
        # Simple split by comma, then normalize
        authors = set()
        for author in authors_str.split(','):
            author = author.strip().lower()
            if author:
                authors.add(author)
        return authors


def compute_jaccard_similarity(authors1, authors2):

    if not authors1 or not authors2:
        return 0.0

    intersection = len(authors1 & authors2)
    union = len(authors1 | authors2)

    return intersection / union if union > 0 else 0.0


def compute_similarity_matrix(papers):

    n = len(papers)
    similarity_matrix = np.zeros((n, n))

    print(f"Computing author sets for {n} papers...")
    author_sets = [parse_authors(paper) for paper in papers]

    print(f"Computing pairwise similarities...")
    for i in range(n):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n} papers...")

        for j in range(i + 1, n):
            sim = compute_jaccard_similarity(author_sets[i], author_sets[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    print(f"Similarity matrix computed\n")
    return similarity_matrix, author_sets


def create_graph(papers, similarity_matrix, k=5):

    n = len(papers)
    G = nx.Graph()

    print(f"Creating graph with K={k} connections per node...")

    # Add nodes with attributes
    for i, paper in enumerate(papers):
        G.add_node(
            paper['id'],
            title=paper['title'],
            authors=paper['authors'],
            categories=paper['categories'],
            index=i
        )

    # For each paper, connect to top K most similar papers
    edges_added = 0
    for i in range(n):
        # Get similarities for paper i (excluding self)
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1  # Exclude self

        # Find indices of top K most similar papers
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        # Add edges to top K similar papers
        for j in top_k_indices:
            if similarities[j] > 0:  # Only add edge if there's actual similarity
                # Add edge with similarity weight
                G.add_edge(
                    papers[i]['id'],
                    papers[j]['id'],
                    weight=float(similarities[j])
                )
                edges_added += 1

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}\n")

    return G


def save_graph(G, output_path):

    print(f"Saving graph to {output_path}...")
    nx.write_graphml(G, output_path)
    print(f"Graph saved successfully\n")


def print_graph_stats(G):

    print("=" * 60)
    print("GRAPH STATISTICS")
    print("=" * 60)
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")

    # Check connectivity
    if nx.is_connected(G):
        print(f"Graph is connected")
        print(f"Diameter: {nx.diameter(G)}")
    else:
        components = list(nx.connected_components(G))
        print(f"Graph has {len(components)} connected components")
        print(f"Largest component size: {len(max(components, key=len))}")

    # Degree distribution
    degrees = [d for n, d in G.degree()]
    print(f"Min degree: {min(degrees)}")
    print(f"Max degree: {max(degrees)}")
    print(f"Median degree: {np.median(degrees):.2f}")

    print("=" * 60)


def visualize_graph(G, output_file='author_network.html'):
    print(f"\nCreating interactive visualization...")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Create PyVis network
    try:
        net = Network(height='900px', width='100%', notebook=False,
                      bgcolor='#1a1a1a', font_color='white',
                      cdn_resources='in_line')
    except Exception:
        # Fallback to simpler initialization
        net = Network(height='900px', width='100%')

    # Physics settings for nice layout
    try:
        net.barnes_hut(
            gravity=-5000,
            central_gravity=0.3,
            spring_length=150,
            spring_strength=0.001,
            damping=0.09
        )
    except Exception as e:
        print(f"Warning: Could not set physics settings: {e}")

    # Add nodes with styling
    print("Adding nodes to visualization...")
    for node_id, node_data in G.nodes(data=True):
        title = node_data.get('title', 'Unknown')
        authors = node_data.get('authors', 'Unknown')
        categories = node_data.get('categories', 'Unknown')
        degree = G.degree(node_id)

        # Truncate long text for label
        label = title[:40] + '...' if len(title) > 40 else title

        # Create hover text with full info
        hover_text = (
            f"<b>{title}</b><br>"
            f"Authors: {authors[:100]}{'...' if len(authors) > 100 else ''}<br>"
            f"Categories: {categories}<br>"
            f"Connections: {degree}"
        )

        # Color based on degree (green = more connections)
        color_hue = min(degree * 10, 120)  # Cap at green

        net.add_node(
            node_id,
            label=label,
            title=hover_text,
            size=10 + degree * 2,
            color=f'hsl({color_hue}, 70%, 50%)'
        )

    # Add edges
    print("Adding edges to visualization...")
    for edge in G.edges(data=True):
        weight = float(edge[2].get('weight', 0.1))
        # Scale edge weight for visibility
        net.add_edge(edge[0], edge[1], value=weight * 10)

    # Save visualization
    print(f"Saving visualization to {output_file}...")
    try:
        # Generate HTML and write with UTF-8 encoding to handle special characters
        html = net.generate_html()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✓ Visualization saved to {output_file}")
        print(f"  Open this file in your browser to interact with the network!")
    except Exception as e:
        print(f"Warning: Could not save visualization: {e}")
        print(f"The graph was still saved to GraphML format successfully.")


def main():
    # Paths
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'arxiv-metadata-oai-snapshot.json'
    output_path = Path(__file__).parent.parent / 'data' / 'author_similarity_graph.graphml'
    viz_output = Path(__file__).parent.parent / 'visualizations' / 'author_network.html'

    # Parameters
    num_papers = 10000
    k = 10  # Number of connections per paper

    # Load papers
    papers = load_papers(data_path, num_papers)

    # Compute similarity matrix
    similarity_matrix, author_sets = compute_similarity_matrix(papers)

    # Create graph
    G = create_graph(papers, similarity_matrix, k)

    # Print statistics
    print_graph_stats(G)

    # Save graph
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_graph(G, output_path)

    print(f"\nDone! Graph saved to: {output_path}")
    print(f"You can load this graph in NetworkX, Gephi, or Cytoscape for analysis.")

    # Create visualization
    viz_output.parent.mkdir(parents=True, exist_ok=True)
    visualize_graph(G, output_file=str(viz_output))

    print("\n" + "=" * 60)
    print("ALL COMPLETE!")
    print("=" * 60)
    print(f"\nGraph file: {output_path}")
    print(f"Visualization: {viz_output}")
    print("\nNext steps:")
    print("  1. Open the HTML file in your browser to explore the network")
    print("  2. Use the graph file for further analysis")
    print("  3. Try increasing num_papers or k for more connections")


if __name__ == '__main__':
    main()