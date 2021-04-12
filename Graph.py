import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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
        self.partition_S = None
        self.partition_T = None
        self.partition_S_labels = None
        self.partition_T_labels = None


    def show(self):
        scaled_weights = np.log(np.log(np.log(np.array(self.W) + 1) + 1) + 1)
        scaled_weights = scaled_weights / np.max(scaled_weights) * 5

        nx.draw_networkx_nodes(self.G, self.positions_dict, node_size=1, node_color=self.colors)
        #nx.draw_networkx_labels(self.G, self.positions_dict, font_size=6, labels=self.labels_dict)
        nx.draw_networkx_edges(self.G, self.positions_dict, edgelist=self.E, width=scaled_weights, edge_color=self.edge_colors)
        plt.show()

    def min_cut(self, start_from_previous_graph=True):
        R = nx.algorithms.flow.boykov_kolmogorov(self.G, self.vertices_dict['S'], self.vertices_dict['T'], capacity='weight', residual=self.R if start_from_previous_graph else None)
        self.R = R
        source_tree, _ = R.graph["trees"]
        partition_1 = set(source_tree)
        partition_2 = set(self.G) - partition_1

        if (self.vertices_dict['S'] in partition_1):
            self.partition_S = partition_1
            self.partition_T = partition_2
        else:
            self.partition_S = partition_2
            self.partition_T = partition_1

        self.partition_S_labels = [self.labels_dict[v] for v in self.partition_S]
        self.partition_T_labels = [self.labels_dict[v] for v in self.partition_T]

        E_new = []
        W_new = []
        edge_colors_new = []
        for i in range(len(self.E)):
            edge = self.E[i]
            weight = self.W[i]
            edge_color = self.edge_colors[i]

            if (edge[0] in partition_1 and edge[1] in partition_1) or \
               (edge[0] not in partition_1 and edge[1] not in partition_1):
               E_new.append(edge)
               W_new.append(weight)
               edge_colors_new.append(edge_color)


        self.E = E_new
        self.W = W_new
        self.edge_colors = edge_colors_new




tSeeds = [(1,2), (3,4), (3,3)]
sSeeds = [(3,1), (3,2)]

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

    labels = list([(i%imgw, i//imgw) for i in range(imgw * imgh)]) + ['S', 'T']
    labels_dict = dict(zip(vertices, labels))

    positions = [((i//imgw), -(i%imgw+2)) for i in range(imgw * imgh)] + \
                [((imgh-1)/2, 0), ((imgh-1)/2, -(imgw+3))]

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

    colors = ['#add8e6'] * (imgw * imgh) + ['#ffcccb', '#ffcccb']
    edge_colors = ['red' if edge[0] in terminal_vertices or edge[1] in terminal_vertices else 'blue' for edge in edges]

    G = Graph(vertices, edges, weights, labels, positions, colors, edge_colors)
    G.show()

    G.min_cut(start_from_previous_graph=False)
    G.show()
