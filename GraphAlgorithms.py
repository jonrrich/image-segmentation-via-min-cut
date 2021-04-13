import numpy as np
import cv2 as cv
from Graph import *
import matplotlib.pyplot as plt



class GraphAlgorithms:

    def __init__(self, frames, tSeeds, sSeeds, lmbda=50, R_bins=5):

        self.lmbda = lmbda

        self.tSeeds = set(tSeeds) #background
        self.sSeeds = set(sSeeds) #object

        self.frames = frames

        self.B = dict()
        if frames.shape[0]>1:
            self.construct_B_vid()

            tVals = [frames[i[2], i[1], i[0]] for i in self.tSeeds]
            sVals = [frames[i[2], i[1], i[0]] for i in self.sSeeds]

        else:
            img = frames[0]
            self.img = img
            self.construct_B_img(img)

            tVals = [img[i[1], i[0]] for i in self.tSeeds]
            sVals = [img[i[1], i[0]] for i in self.sSeeds]


        self.tHist, self.bin_edges = np.histogram(tVals,bins=R_bins,range=(0,255))
        self.tHist = self.tHist / sum(self.tHist)

        self.sHist, bin_edges = np.histogram(sVals,bins=self.bin_edges)
        self.sHist = self.sHist / sum(self.sHist)

        print("Creating graph")
        self.G = self.create_graph()


    def construct_B_img(self,img):
        laplacian = cv.Laplacian(img,cv.CV_64F)

        for x in range(laplacian.shape[1]):
            for y in range(laplacian.shape[0]):
                edges = []
                if y+1 < laplacian.shape[0]:
                    edges.append([(x, y), (x, y+1)])
                if x+1 < laplacian.shape[1]:
                    edges.append([(x, y), (x+1, y)])
                for edge in edges:
                    laplacian_edge = (abs(laplacian[edge[0][1], edge[0][0]]) + abs(laplacian[edge[1][1], edge[1][0]]))/2
                    self.B[frozenset(edge)] = -1 if laplacian_edge==0 else 1/laplacian_edge

        max_B = max(self.B.values())+1
        for i in self.B:
            if self.B[i] == -1:
                self.B[i] = max_B

        self.K = max_B+1


    def construct_B_vid(self):
        laplacian = []
        for z in range(self.frames.shape[0]):
            img = self.frames[z]
            laplacian.append(cv.Laplacian(img,cv.CV_64F))

        laplacian = np.array(laplacian)
        print(laplacian.shape)

        for z in range(self.frames.shape[0]):
            for x in range(laplacian.shape[2]):
                for y in range(laplacian.shape[1]):
                    edges = []
                    if y+1 < laplacian.shape[1]:
                        edges.append([(x, y, z), (x, y+1, z)])
                    if x+1 < laplacian.shape[2]:
                        edges.append([(x, y, z), (x+1, y, z)])
                    if z>0:
                        edges.append([(x, y, z), (x, y, z-1)])

                    for edge in edges:
                        laplacian_edge = (abs(laplacian[edge[0][2], edge[0][1], edge[0][0]]) + abs(laplacian[edge[1][2], edge[1][1], edge[1][0]]))/2
                        self.B[frozenset(edge)] = -1 if laplacian_edge==0 else 1/laplacian_edge

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
                prob = 1e-6 if self.sHist[idx]==0 else self.sHist[idx]
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
                prob = 1e-6 if self.tHist[idx]==0 else self.tHist[idx]
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
                weights.append(self.tLinkWeight(pixel, self.frames[pixel[2], pixel[1],pixel[0]], labels_dict[edge[1]]))

            # n-links vertically
            if i%(imgw*imgh) < imgw * (imgh-1):
                edge = (i, i+imgw)
                edges.append(edge)
                edge_colors.append('black')
                weights.append(self.nLinkWeight(labels_dict[edge[0]], labels_dict[edge[1]]))

            # n-links between frames
            if i+imgw*imgh < imgw * imgh * imgn:
                edge = (i, i+imgw*imgh)
                edges.append(edge)
                edge_colors.append('grey')
                weights.append(self.nLinkWeight(labels_dict[edge[0]], labels_dict[edge[1]]))

        colors = ['#add8e6'] * (imgw * imgh * imgn) + ['#ffcccb', '#ffcccb']

        G = Graph(vertices, edges, weights, labels, positions, colors, edge_colors)

        return G


if __name__ == "__main__":
    main()
