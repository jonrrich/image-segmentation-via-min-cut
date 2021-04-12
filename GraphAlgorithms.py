import numpy as np
import cv2 as cv



class GraphAlgorithms:

    def __init__(self, img, tSeeds, sSeeds, lmbda=1, R_bins=50):
        laplacian = cv.Laplacian(img,cv.CV_64F)
        lapl_padded = np.pad(laplacian,1,mode='edge')

        self.B = dict()

        neighbors = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
        for x in range(laplacian.shape[0]):
            for y in range(laplacian.shape[1]):
                for n in neighbors:

                    if {(x,y),n} not in self.B:
                        diff = abs(lapl_padded[x+1,y+1] - lapl_padded[n[0]+1,n[1]+1])

                        if diff==0:
                            self.B[{(x,y),n}] = -1;
                        else:
                            self.B[{(x,y),n}] = 1/diff

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


        self.G = create_graph(self, img.shape[0], img.shape[1])


    def tLinkWeight(pixel, intensity, terminal):
        if terminal == 'T':
            if pixel in tSeeds:
                return self.K
            elif pixel in sSeeds:
                return 0
            else:
                hist, bin_edges = np.histogram(intensity, self.bin_edges)
                idx = np.where(hist==1)[0][0]
                return -np.log(self.sHist[idx])*self.lmbda


        if terminal == 'S':
            if pixel in sSeeds:
                return self.K
            elif pixel in tSeeds:
                return 0
            else:
                hist, bin_edges = np.histogram(intensity, self.bin_edges)
                idx = np.where(hist==1)[0][0]
                return -np.log(self.tHist[idx])*self.lmbda


    def nLinkWeight(self, pixel1, pixel2):
        return self.B[{pixel1,pixel2}]



    def create_graph(self, imgw, imgh):
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

        return G


if __name__ == "__main__":
    main()
