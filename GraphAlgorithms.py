import numpy as np
import cv2 as cv
from Graph import *



class GraphAlgorithms:

    def __init__(self, img, tSeeds, sSeeds, lmbda=3, R_bins=50):
        self.img = img

        laplacian = cv.Laplacian(img,cv.CV_64F)
        lapl_padded = np.pad(laplacian,1,mode='edge')

        self.B = dict()

        for x in range(laplacian.shape[0]):
            for y in range(laplacian.shape[1]):
                neighbors = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
                for n in neighbors:
                    pair = frozenset({(x,y),n})
                    if pair not in self.B:
                        diff = abs(lapl_padded[x+1,y+1] - lapl_padded[n[0]+1,n[1]+1])

                        if diff==0:
                            self.B[pair] = -1;
                        else:
                            self.B[pair] = 1/diff

        max_B = max(self.B.values())+1
        for i in self.B:
            if self.B[i] == -1:
                self.B[i] = max_B

        self.K = max_B+1
        self.lmbda = lmbda

        self.tSeeds = tSeeds #background
        self.sSeeds = sSeeds #object

        tVals = [img[i] for i in self.tSeeds]
        sVals = [img[i] for i in self.sSeeds]

        self.tHist, self.bin_edges = np.histogram(tVals,bins=R_bins,range=(0,255))
        self.tHist = self.tHist / sum(self.tHist)

        self.sHist, bin_edges = np.histogram(sVals,bins=self.bin_edges)
        self.sHist = self.sHist / sum(self.sHist)

        print("Creating graph")
        self.G = self.create_graph(img.shape[0], img.shape[1])


    def tLinkWeight(self, pixel, intensity, terminal):
        if terminal == 'T':
            if pixel in self.tSeeds:
                return self.K
            elif pixel in self.sSeeds:
                return 0
            else:
                hist, bin_edges = np.histogram(intensity, self.bin_edges)
                idx = np.where(hist==1)[0][0]
                return -np.log(self.sHist[idx]+1)*self.lmbda

        # FIX SO WE ITERATE THRU SEEDS INSTEAD OF SEARCHING LISTS
        if terminal == 'S':
            if pixel in self.sSeeds:
                return self.K
            elif pixel in self.tSeeds:
                return 0
            else:
                hist, bin_edges = np.histogram(intensity, self.bin_edges)
                idx = np.where(hist==1)[0][0]
                return -np.log(self.tHist[idx]+1)*self.lmbda


    def nLinkWeight(self, pixel1, pixel2):
        return self.B[frozenset({pixel1,pixel2})]



    def create_graph(self, imgw, imgh):
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
                pixel = labels_dict[edge[0]]
                weights.append(self.tLinkWeight(pixel, self.img[pixel], labels_dict[edge[1]]))

            # n-links horizontally
            if (i+1) < imgw * imgh and (i+1) % imgw != 0:
                edge = (i, i+1)
                edges.append(edge)
                weights.append(self.nLinkWeight(labels_dict[edge[0]], labels_dict[edge[1]]))

            # n-links vertically
            if (i+imgw) < imgw * imgh:
                edge = (i, i+imgw)
                edges.append(edge)
                weights.append(self.nLinkWeight(labels_dict[edge[0]], labels_dict[edge[1]]))

        colors = ['#add8e6'] * (imgw * imgh) + ['#ffcccb', '#ffcccb']
        edge_colors = ['red' if labels[edge[0]] == 'S' or labels[edge[1]] == 'S'
                        else 'blue' if labels[edge[0]] == 'T' or labels[edge[1]] == 'T'
                        else 'black' for edge in edges]

        G = Graph(vertices, edges, weights, labels, positions, colors, edge_colors)

        return G


if __name__ == "__main__":
    main()
