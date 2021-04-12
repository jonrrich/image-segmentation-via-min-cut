import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
        print(pts)

        if pts[0][0] < pts[1][0]:
            min_x = pts[0][0]
            max_x = pts[1][0]
        else:
            min_x = pts[1][0]
            max_x = pts[0][0]

        if pts[0][1] < pts[1][1]:
            min_y = pts[0][1]
            max_y = pts[1][1]
        else:
            min_y = pts[1][1]
            max_y = pts[0][1]

        region = [int(min_x),int(max_x),int(min_y),int(max_y)]
        all_regions.append(region)

        # draw rectangle
        color = 'r' if region_type=="object" else 'b'
        plt.gca().add_patch(Rectangle((min_x,min_y),max_x-min_x,max_y-min_y,linewidth=1,edgecolor=color,facecolor=color))

        message('Mouse click to select another region\nKey click to move on')

        if plt.waitforbuttonpress():
            print('returning')
            return all_regions


def main():
    img = plt.imread('images/dog.jpg')

    obj_regions = select_regions(img,"object")
    background_regions = select_regions(img,"background")

    obj_seeds = set([(x,y) for reg in obj_regions for x in range(reg[0],reg[1]+1) for y in range(reg[2],reg[3]+1)])
    back_seeds = set([(x,y) for reg in background_regions for x in range(reg[0],reg[1]+1) for y in range(reg[2],reg[3]+1)])



if __name__ == "__main__":
    main()
