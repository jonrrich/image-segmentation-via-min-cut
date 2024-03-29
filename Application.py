import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from Image import *
from GraphAlgorithms import *
import networkx as nx
import os
import math
import time


# Some code from tutorial: https://matplotlib.org/stable/gallery/event_handling/ginput_manual_clabel_sgskip.html#sphx-glr-gallery-event-handling-ginput-manual-clabel-sgskip-py

def message(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()


def select_regions(img,region_type):
    plt.imshow(img)
    all_regions = []

    while True:
        message("Define rectangular " + region_type + " region")
        pts = plt.ginput(n=2,timeout=-1)

        for i in (0,1):
            pts[i] = [pts[i][0]+.5, pts[i][1]+.5]

        if pts[0][0] > pts[1][0]:
            pts[0][0] = math.ceil(pts[0][0])-1e-6
            pts[1][0] = math.floor(pts[1][0])
        else:
            pts[0][0] = math.floor(pts[0][0])
            pts[1][0] = math.ceil(pts[1][0])-1e-6

        if pts[0][1] > pts[1][1]:
            pts[0][1] = math.ceil(pts[0][1])-1e-6
            pts[1][1] = math.floor(pts[1][1])
        else:
            pts[0][1] = math.floor(pts[0][1])
            pts[1][1] = math.ceil(pts[1][1])-1e-6

        print(pts)



        if pts[0][0] < pts[1][0]:
            min_y = pts[0][0]
            max_y = pts[1][0]
        else:
            min_y = pts[1][0]
            max_y = pts[0][0]

        if pts[0][1] < pts[1][1]:
            min_x = pts[0][1]
            max_x = pts[1][1]
        else:
            min_x = pts[1][1]
            max_x = pts[0][1]

        region = [int(min_x),int(max_x),int(min_y),int(max_y)]
        all_regions.append(region)

        # draw rectangle
        color = 'r' if region_type=="object" else 'b'
        plt.gca().add_patch(Rectangle((min_y-.5,min_x-.5),max_y-min_y,max_x-min_x,linewidth=1,edgecolor=color,facecolor=color))

        message('Mouse click to select another region\nKey click to move on')

        if plt.waitforbuttonpress():
            return all_regions


def test_lambda():
    Img = Image('images/dog2.jpg')
    img = Img.img

    obj_regions = select_regions(img,"object")
    background_regions = select_regions(img,"background")
    plt.close()

    obj_seeds = set([(x,y,0) for reg in obj_regions for y in range(reg[0],reg[1]+1) for x in range(reg[2],reg[3]+1)])
    back_seeds = set([(x,y,0) for reg in background_regions for y in range(reg[0],reg[1]+1) for x in range(reg[2],reg[3]+1)])
    print("Seeds created")

    lm_list = np.array([0., 0.1, 0.5, 1, 3, 6])
    for lm in lm_list:
        GraphAlgos = GraphAlgorithms(np.array([Img.gray_img]), back_seeds, obj_seeds, lmbda=lm)
        G = GraphAlgos.G
        print("Graph made")
        #G.show()
        G.min_cut()
        print("Min cut found")
        #G.show()
        plt.close()

        segmented = Img.segmentation(G.partition_S_labels,process=True)
        masked = Img.apply_mask(segmented)
        plt.imshow(masked)
        plt.title("Lambda = " + str(lm), fontsize=16)
        plt.draw()

        plt.show()
        plt.close()

def run_img():
    Img = Image('images/dog2.jpg')
    img = Img.img

    obj_regions = select_regions(img,"object")
    background_regions = select_regions(img,"background")
    plt.close()

    obj_seeds = set([(x,y,0) for reg in obj_regions for y in range(reg[0],reg[1]+1) for x in range(reg[2],reg[3]+1)])
    back_seeds = set([(x,y,0) for reg in background_regions for y in range(reg[0],reg[1]+1) for x in range(reg[2],reg[3]+1)])

    print("Seeds created")

    GraphAlgos = GraphAlgorithms(np.array([Img.gray_img]), back_seeds, obj_seeds,lmbda=0.1)
    G = GraphAlgos.G
    print("Graph made")


    start_time = time.time()
    print("Start",start_time)

    G.min_cut(flow_func=nx.algorithms.flow.preflow_push)
    print("Min cut found")

    end_time = time.time()
    print("End",end_time)
    print("Preflow Push Time: ", end_time-start_time)


    """
    start_time = time.time()
    print("Start",start_time)

    G.min_cut(flow_func=nx.algorithms.flow.shortest_augmenting_path)
    print("Min cut found")

    end_time = time.time()
    print("End",end_time)
    print("Shortest Augmenting Path Time: ", end_time-start_time)
    """


    segmented = Img.segmentation(G.partition_S_labels)
    masked = Img.apply_mask(segmented)
    plt.imshow(masked)
    plt.show()
    plt.close()


    segmented = Img.segmentation(G.partition_S_labels,process=True)
    masked = Img.apply_mask(segmented)
    plt.imshow(masked)
    plt.show()
    plt.close()

    segmented = Img.segmentation(G.partition_S_labels, process=True)
    masked = Img.apply_mask(segmented, back_img=True)
    plt.imshow(masked)
    plt.show()
    plt.close()

    print("ALGO TIME")
    print(end_time-start_time)


def run_video():
    dir = 'walking_man'
    num_frames = len([f for f in os.listdir(dir) if f[0]!='.'])
    Imgs = []

    obj_seeds = []
    back_seeds = []
    frames = []
    for frame_idx in range(num_frames):
        name = dir + "/frame"+str(frame_idx+1)+".jpg"

        Img = Image(name)
        Imgs.append(Img)
        img = Img.img
        frames.append(Img.gray_img)

        plt.imshow(img)
        message('Frame '+str(frame_idx)+', Mouse click to select seeds\nKey click for next frame')
        if plt.waitforbuttonpress():
            continue

        obj_regions = select_regions(img,"object")
        background_regions = select_regions(img,"background")
        plt.close()

        obj_seeds += set([(x,y,frame_idx) for reg in obj_regions for y in range(reg[0],reg[1]+1) for x in range(reg[2],reg[3]+1)])
        back_seeds += set([(x,y,frame_idx) for reg in background_regions for y in range(reg[0],reg[1]+1) for x in range(reg[2],reg[3]+1)])

        print(obj_seeds)
        print(back_seeds)

    print("Seeds created")
    start_time = time.time()

    GraphAlgos = GraphAlgorithms(np.array(frames), back_seeds, obj_seeds, lmbda=0.1, R_bins=40)

    G = GraphAlgos.G
    print("Graph made")
    plt.close()
    #G.show()
    G.min_cut()
    print("Min cut found")
    #G.show()
    #plt.close()

    end_time = time.time()

    print("Runtime: ", end_time-start_time)


    partition_labels = np.array(G.partition_S_labels)
    for z in range(num_frames):
        partition = [(i[0],i[1]) for i in partition_labels if i!='S' and i[2]==z]

        segmented = Imgs[z].segmentation(partition)
        masked = Imgs[z].apply_mask(segmented)

        message('Frame '+str(z))
        plt.imshow(masked)

        plt.show()
        plt.close()


        segmented = Imgs[z].segmentation(partition,process=True,iterations=1)
        masked = Imgs[z].apply_mask(segmented)

        message('Frame '+str(z)+'\nNoise reduction')
        plt.imshow(masked)

        plt.show()
        plt.close()


if __name__ == "__main__":
    run_img()
    #run_video()
    #test_lambda()
