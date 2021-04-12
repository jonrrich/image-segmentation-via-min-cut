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

        self.labels_dict = dict(zip(vertices, labels))
        self.positions_dict = dict(zip(vertices, positions))


    def show(self):
        nx.draw_networkx_nodes(self.G, self.positions_dict, node_size=200, node_color=self.colors)
        nx.draw_networkx_labels(self.G, self.positions_dict, font_size=10, labels=self.labels_dict)
        nx.draw_networkx_edges(self.G, self.positions_dict, edgelist=self.E, width=self.W, edge_color=self.edge_colors)
        plt.show()


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
    terminal_vertices = [vertices[-2], vertices[-1]]
    labels = list(range(imgw * imgh)) + ['S', 'T']

    positions = [((i%imgw), -(i//imgw+2)) for i in range(imgw * imgh)] + \
                [((imgw-1)/2, 0), ((imgw-1)/2, -(imgh+3))]

    edges = []
    weights = []
    for i in range(imgw * imgh):
        # t-links
        for terminal_vertex in terminal_vertices:
            edge = (i, terminal_vertex)
            edges.append(edge)
            weights.append(tLinkWeight(labels[edge[0]], labels[edge[1]]))

        # n-links horizontally
        if (i+1) < imgw * imgh and (i+1) % imgw != 0:
            edge = (i, i+1)
            edges.append(edge)
            weights.append(nLinkWeight(labels[edge[0]], labels[edge[1]]))

        # n-links vertically
        if (i+imgw) < imgw * imgh:
            edge = (i, i+imgw)
            edges.append(edge)
            weights.append(nLinkWeight(labels[edge[0]], labels[edge[1]]))

    colors = ['blue'] * (imgw * imgh) + ['red', 'red']
    edge_colors = ['red' if edge[0] in terminal_vertices or edge[1] in terminal_vertices else 'blue' for edge in edges]

    G = Graph(vertices, edges, weights, labels, positions, colors, edge_colors)
    G.show()
