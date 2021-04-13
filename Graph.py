import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

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
        scaled_weights = np.log(np.log(np.array(self.W) + 1) + 1)
        scaled_weights = scaled_weights / np.max(scaled_weights) * 5

        nx.draw_networkx_nodes(self.G, self.positions_dict, node_size=5, node_color=self.colors)
        # nx.draw_networkx_labels(self.G, self.positions_dict, font_size=6, labels=self.labels_dict)
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
            return 100
    if terminal == 'S':
        if pixel in sSeeds:
            return 100
    return 1


def nLinkWeight(pixel1, pixel2):
    return random.choice([1, 2, 3])



if __name__ == "__main__":
    imgw = 3
    imgh = 3
    imgn = 3

    vertices = list(range(imgw * imgh * imgn + 2))
    pixel_vertices = vertices[0:-2]
    terminal_vertices = vertices[-2:]

    labels = list([((i%(imgw*imgh))%imgw, (i%(imgw*imgh))//imgw, i//(imgw*imgh)) for i in range(imgw * imgh * imgn)]) + ['S', 'T']
    labels_dict = dict(zip(vertices, labels))

    positions = [( ((i%(imgw*imgh))%imgw) + .2*(i//(imgw*imgh)), -((i%(imgw*imgh))//imgw+2) + .2*(i//(imgw*imgh)) ) for i in range(imgw * imgh * imgn)] + \
                [((imgw-1)/2 + .1*(imgn-1), 0), ((imgw-1)/2 + .1*(imgn-1), -(imgh+3))]

    edges = []
    edge_colors = []
    weights = []
    for i in pixel_vertices:
        # t-links
        for terminal_vertex in terminal_vertices:
            edge = (i, terminal_vertex)
            edges.append(edge)
            edge_colors.append('red' if labels_dict[terminal_vertex] == 'S' else 'blue')
            pixel = labels_dict[edge[0]]
            weights.append(tLinkWeight(pixel, labels_dict[edge[1]]))

        # n-links horizontally
        if (i%(imgw*imgh))%imgw < imgw - 1:
            edge = (i, i+1)
            edges.append(edge)
            edge_colors.append('black')
            weights.append(nLinkWeight(labels_dict[edge[0]], labels_dict[edge[1]]))

        # n-links vertically
        if i%(imgw*imgh) < imgw * (imgh-1):
            edge = (i, i+imgw)
            edges.append(edge)
            edge_colors.append('black')
            weights.append(nLinkWeight(labels_dict[edge[0]], labels_dict[edge[1]]))

        # n-links between frames
        if i+imgw*imgh < imgw * imgh * imgn:
            edge = (i, i+imgw*imgh)
            edges.append(edge)
            edge_colors.append('grey')
            weights.append(nLinkWeight(labels_dict[edge[0]], labels_dict[edge[1]]))

    colors = ['#add8e6'] * (imgw * imgh * imgn) + ['#ffcccb', '#ffcccb']

    G = Graph(vertices, edges, weights, labels, positions, colors, edge_colors)

    G.show()
