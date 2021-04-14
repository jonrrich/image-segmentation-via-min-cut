import numpy as np
import cv2 as cv
from Graph import *
import matplotlib.pyplot as plt



class GraphAlgorithms:

    def __init__(self, frames, tSeeds, sSeeds, lmbda=0.1, R_bins=10):

        self.lmbda = lmbda

        self.tSeeds = set(tSeeds) #background
        self.sSeeds = set(sSeeds) #object

        self.frames = frames

        self.B = dict()

        if frames.shape[0]>1:
            self.construct_B_vid()
        else:
            self.construct_B_img()

        tVals = [frames[i[2], i[1], i[0]] for i in self.tSeeds]
        sVals = [frames[i[2], i[1], i[0]] for i in self.sSeeds]

        self.tHist, self.bin_edges = np.histogram(tVals,bins=R_bins,range=(0,255))
        self.tHist = self.tHist / sum(self.tHist)

        self.sHist, bin_edges = np.histogram(sVals,bins=self.bin_edges)
        self.sHist = self.sHist / sum(self.sHist)

        print("Creating graph")
        self.G = self.create_graph()


    def construct_B_vid(self):

        axis0 = np.gradient(np.gradient(self.frames,axis=0),axis=0)
        axis1 = np.gradient(np.gradient(self.frames,axis=1),axis=1)
        axis2 = np.gradient(np.gradient(self.frames,axis=2),axis=2)

        for z in range(self.frames.shape[0]):
            for x in range(self.frames.shape[2]):
                for y in range(self.frames.shape[1]):
                    if y+1 < self.frames.shape[1]:
                        edge = [(x, y, z), (x, y+1, z)]
                        laplacian_edge = (abs(axis1[edge[0][2], edge[0][1], edge[0][0]]) + abs(axis1[edge[1][2], edge[1][1], edge[1][0]]))/2

                        if laplacian_edge==0:
                            self.B[frozenset(edge)] = -1
                        else:
                            self.B[frozenset(edge)] = 6 if (1/laplacian_edge)>6 else 1/laplacian_edge

                    if x+1 < self.frames.shape[2]:
                        edge = [(x, y, z), (x+1, y, z)]
                        laplacian_edge = (abs(axis2[edge[0][2], edge[0][1], edge[0][0]]) + abs(axis2[edge[1][2], edge[1][1], edge[1][0]]))/2

                        if laplacian_edge==0:
                            self.B[frozenset(edge)] = -1
                        else:
                            self.B[frozenset(edge)] = 6 if (1/laplacian_edge)>6 else 1/laplacian_edge

                    if z>0:
                        edge = [(x, y, z), (x, y, z-1)]
                        laplacian_edge = (abs(axis0[edge[0][2], edge[0][1], edge[0][0]]) + abs(axis0[edge[1][2], edge[1][1], edge[1][0]]))/2

                        if laplacian_edge==0:
                            self.B[frozenset(edge)] = -1
                        else:
                            self.B[frozenset(edge)] = 6 if (1/laplacian_edge)>6 else 1/laplacian_edge

        max_B = max(self.B.values())+1
        for i in self.B:
            if self.B[i] == -1:
                self.B[i] = max_B

        self.K = max_B+1


    def construct_B_img(self):

        frames = self.frames[0]

        axis0 = np.gradient(np.gradient(frames,axis=0),axis=0)
        axis1 = np.gradient(np.gradient(frames,axis=1),axis=1)

        for x in range(frames.shape[1]):
            for y in range(frames.shape[0]):
                if y+1 < frames.shape[0]:
                    edge = [(x, y,0), (x, y+1,0)]
                    laplacian_edge = (abs(axis0[edge[0][1], edge[0][0]]) + abs(axis0[edge[1][1], edge[1][0]]))/2

                    if laplacian_edge==0:
                        self.B[frozenset(edge)] = -1
                    else:
                        self.B[frozenset(edge)] = 6 if (1/laplacian_edge)>6 else 1/laplacian_edge

                if x+1 < frames.shape[1]:
                    edge = [(x, y,0), (x+1, y,0)]
                    laplacian_edge = (abs(axis1[edge[0][1], edge[0][0]]) + abs(axis1[edge[1][1], edge[1][0]]))/2

                    if laplacian_edge==0:
                        self.B[frozenset(edge)] = -1
                    else:
                        self.B[frozenset(edge)] = 6 if (1/laplacian_edge)>6 else 1/laplacian_edge


        max_B = max(self.B.values())+1
        for i in self.B:
            if self.B[i] == -1:
                self.B[i] = max_B

        self.K = max_B+1


    def tLinkWeight(self, pixel, intensity, terminal):
        if terminal == 'T':
            if pixel in self.tSeeds:
                return self.K
            elif pixel in self.sSeeds:
                return 0
            else:
                hist, bin_edges = np.histogram(intensity, self.bin_edges)
                idx = -np.where(hist==1)[0][0]
                prob = 0.01 if self.sHist[idx]==0 else self.sHist[idx]
                R = -np.log(prob)*self.lmbda
                return R

        if terminal == 'S':
            if pixel in self.sSeeds:
                return self.K
            elif pixel in self.tSeeds:
                return 0
            else:
                hist, bin_edges = np.histogram(intensity, self.bin_edges)
                idx = np.where(hist==1)[0][0]
                prob = 0.01 if self.tHist[idx]==0 else self.tHist[idx]
                R = -np.log(prob)*self.lmbda
                return R


    def nLinkWeight(self, pixel1, pixel2):
        return self.B[frozenset({pixel1,pixel2})]


    def create_graph(self):
        imgw = self.frames.shape[2]
        imgh = self.frames.shape[1]
        imgn = self.frames.shape[0]

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
        t_weights = []
        n_weights = []
        for i in pixel_vertices:
            # t-links
            for terminal_vertex in terminal_vertices:
                edge = (i, terminal_vertex)
                edges.append(edge)
                edge_colors.append('red' if labels_dict[terminal_vertex] == 'S' else 'blue')
                pixel = labels_dict[edge[0]]

                w = self.tLinkWeight(pixel, self.frames[pixel[2], pixel[1],pixel[0]], labels_dict[edge[1]])
                weights.append(w)
                t_weights.append(w)

            # n-links horizontally
            if (i%(imgw*imgh))%imgw < imgw - 1:
                edge = (i, i+1)
                edges.append(edge)
                edge_colors.append('black')

                w = self.nLinkWeight(labels_dict[edge[0]], labels_dict[edge[1]])
                weights.append(w)
                n_weights.append(w)

            # n-links vertically
            if i%(imgw*imgh) < imgw * (imgh-1):
                edge = (i, i+imgw)
                edges.append(edge)
                edge_colors.append('black')

                w = self.nLinkWeight(labels_dict[edge[0]], labels_dict[edge[1]])
                weights.append(w)
                n_weights.append(w)

            # n-links between frames
            if i+imgw*imgh < imgw * imgh * imgn:
                edge = (i, i+imgw*imgh)
                edges.append(edge)
                edge_colors.append('grey')

                w = self.nLinkWeight(labels_dict[edge[0]], labels_dict[edge[1]])
                weights.append(w)
                n_weights.append(w)

        colors = ['#add8e6'] * (imgw * imgh * imgn) + ['#ffcccb', '#ffcccb']

        G = Graph(vertices, edges, weights, labels, positions, colors, edge_colors)

        return G


if __name__ == "__main__":
    main()
