import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, vertices, edges, weights, labels, positions, colors, edge_colors):
        assert len(edges) == len(weights)
        assert len(vertices) == len(labels)
        assert len(vertices) == len(positions)
        assert len(vertices) == len(colors)

        G = nx.Graph()
        for i in range(len(vertices)):
            pos = positions[i]
            G.add_node(vertices[i], x=pos[0], y=pos[1])

        for i in range(len(edges)):
            edge = edges[i]
            G.add_edge(edge[0], edge[1], weight=weights[i])

        self.G = G
        self.V = vertices
        self.E = edges
        self.W = weights
        self.labels = labels
        self.positions = positions
        self.colors = colors
        self.edge_colors = edge_colors

        self.vertices_dict = dict(zip(labels, vertices))
        self.labels_dict = dict(zip(vertices, labels))
        self.positions_dict = dict(zip(vertices, positions))

        self.R = None


    def show(self):
        nx.draw_networkx_nodes(self.G, self.positions_dict, node_size=200, node_color=self.colors)
        nx.draw_networkx_labels(self.G, self.positions_dict, font_size=10, labels=self.labels_dict)
        nx.draw_networkx_edges(self.G, self.positions_dict, edgelist=self.E, width=self.W, edge_color=self.edge_colors)
        plt.show()

    def min_cut(self, start_from_previous_graph=True):
        R = nx.algorithms.flow.boykov_kolmogorov(self.G, self.vertices_dict['S'], self.vertices_dict['T'], capacity='weight', residual=self.R if start_from_previous_graph else None)
        self.R = R
        source_tree, _ = R.graph["trees"]
        source_tree = set(source_tree)

        E_new = []
        W_new = []
        edge_colors_new = []
        for i in range(len(self.E)):
            edge = self.E[i]
            weight = self.W[i]
            edge_color = self.edge_colors[i]

            if (edge[0] in source_tree and edge[1] in source_tree) or \
               (edge[0] not in source_tree and edge[1] not in source_tree):
               E_new.append(edge)
               W_new.append(weight)
               edge_colors_new.append(edge_color)


        self.E = E_new
        self.W = W_new
        self.edge_colors = edge_colors_new




tSeeds = [4, 18, 9]
sSeeds = [6, 11, 15]

def tLinkWeight(pixel, terminal):
    if terminal == 'T':
        if pixel in tSeeds:
            return 10
    if terminal == 'S':
        if pixel in sSeeds:
            return 10
    return 1


def nLinkWeight(pixel1, pixel2):
    return 1



if __name__ == "__main__":
    imgw = 4
    imgh = 5

    vertices = list(range(imgw * imgh + 2))
    pixel_vertices = vertices[0:-2]
    terminal_vertices = vertices[-2:]

    labels = list(range(imgw * imgh)) + ['S', 'T']
    labels_dict = dict(zip(vertices, labels))

    positions = [((i%imgw), -(i//imgw+2)) for i in range(imgw * imgh)] + \
                [((imgw-1)/2, 0), ((imgw-1)/2, -(imgh+3))]

    edges = []
    weights = []
    for i in pixel_vertices:
        # t-links
        for terminal_vertex in terminal_vertices:
            edge = (i, terminal_vertex)
            edges.append(edge)
            weights.append(tLinkWeight(labels_dict[edge[0]], labels_dict[edge[1]]))

        # n-links horizontally
        if (i+1) < imgw * imgh and (i+1) % imgw != 0:
            edge = (i, i+1)
            edges.append(edge)
            weights.append(nLinkWeight(labels_dict[edge[0]], labels_dict[edge[1]]))

        # n-links vertically
        if (i+imgw) < imgw * imgh:
            edge = (i, i+imgw)
            edges.append(edge)
            weights.append(nLinkWeight(labels_dict[edge[0]], labels_dict[edge[1]]))

    colors = ['blue'] * (imgw * imgh) + ['red', 'red']
    edge_colors = ['red' if edge[0] in terminal_vertices or edge[1] in terminal_vertices else 'blue' for edge in edges]

    G = Graph(vertices, edges, weights, labels, positions, colors, edge_colors)
    G.show()

    G.min_cut(start_from_previous_graph=False)
    G.show()