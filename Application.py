import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from Image import *
from GraphAlgorithms import *
import os


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
        #print(pts)

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
        plt.gca().add_patch(Rectangle((min_y,min_x),max_y-min_y,max_x-min_x,linewidth=1,edgecolor=color,facecolor=color))

        message('Mouse click to select another region\nKey click to move on')

        if plt.waitforbuttonpress():
            return all_regions


def test_lambda():
    Img = Image('images/dog2.jpg')
    img = Img.img

    obj_regions = select_regions(img,"object")
    background_regions = select_regions(img,"background")
    plt.close()

    obj_seeds = set([(x,y) for reg in obj_regions for y in range(reg[0],reg[1]+1) for x in range(reg[2],reg[3]+1)])
    back_seeds = set([(x,y) for reg in background_regions for y in range(reg[0],reg[1]+1) for x in range(reg[2],reg[3]+1)])
    print("Seeds created")

    lm_list = np.array([i for i in range(0,600,100)])/10
    for lm in lm_list:
        GraphAlgos = GraphAlgorithms([Img.gray_img], back_seeds, obj_seeds, lmbda=lm)
        G = GraphAlgos.G
        print("Graph made")
        #G.show()
        G.min_cut()
        print("Min cut found")
        #G.show()
        plt.close()

        segmented = Img.segmentation(G.partition_S_labels)
        plt.imshow(segmented)
        plt.title("Lambda = " + str(lm), fontsize=16)
        plt.draw()

        plt.show()
        plt.close()

def run_img():
    Img = Image('images/car3.jpg')
    img = Img.img

    obj_regions = select_regions(img,"object")
    background_regions = select_regions(img,"background")
    plt.close()

    obj_seeds = set([(x,y,0) for reg in obj_regions for y in range(reg[0],reg[1]+1) for x in range(reg[2],reg[3]+1)])
    back_seeds = set([(x,y,0) for reg in background_regions for y in range(reg[0],reg[1]+1) for x in range(reg[2],reg[3]+1)])
    print("Seeds created")

    GraphAlgos = GraphAlgorithms(np.array([Img.gray_img]), back_seeds, obj_seeds)
    G = GraphAlgos.G
    print("Graph made")
    #G.show()
    G.min_cut()
    print("Min cut found")
    #G.show()
    plt.close()

    segmented = Img.segmentation(G.partition_S_labels)
    masked = Img.apply_mask(segmented)

    plt.imshow(masked)

    plt.show()
    plt.close()


def run_video():
    dir = 'walking_man'
    num_frames = len([f for f in os.listdir(dir) if f[0]!='.'])

    obj_seeds = []
    back_seeds = []
    frames = []
    for frame_idx in range(num_frames):
        name = dir + "/frame"+str(frame_idx+1)+".jpg"

        Img = Image(name)
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

    print("Seeds created")

    GraphAlgos = GraphAlgorithms(np.array(frames), back_seeds, obj_seeds)
    G = GraphAlgos.G
    print("Graph made")
    #G.show()
    G.min_cut()
    print("Min cut found")
    #G.show()
    plt.close()

    for z in range(G.partition_S_labels.shape[0]):
        partition = [(i[1],i[2]) for i in G.partition_S_labels() if i[0]==z]
        segmented = Img.segmentation(partition)
        masked = Img.apply_mask(segmented)

        message('Frame '+str(z))
        plt.imshow(masked)

        plt.show()
        plt.close()


if __name__ == "__main__":
    #run_img()
    run_video()
    #test_lambda()
