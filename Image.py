import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


class Image:
    def __init__(self, path):
        self.img = plt.imread(path)
        self.gray_img = 0.2989*self.img[:,:,0] + 0.5870*self.img[:,:,1] + 0.1140*self.img[:,:,2]


    def segmentation(self,obj_pixels,process=False,iterations=2):
        segmented = np.zeros_like(self.gray_img)
        obj_pixels = [p for p in obj_pixels if p != 'S']

        for pixel in obj_pixels:
            segmented[pixel[1],pixel[0]] = 1 if self.gray_img[pixel[1],pixel[0]]>0 else 0

        if process:
            element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))

            # remove holes
            segmented = cv.dilate(segmented, element, iterations=iterations)
            segmented = cv.erode(segmented, element, iterations=iterations)

            # remove noise
            segmented = cv.erode(segmented, element, iterations=iterations)
            segmented = cv.dilate(segmented, element, iterations=iterations)



        return segmented

    def apply_mask(self, binary, back_img=False):
        if back_img:
            masked = np.array(Image('images/space2.jpg').img)
        else:
            masked = np.zeros_like(self.img)


        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                masked[i,j] = self.img[i,j] if binary[i,j]>0 else masked[i,j]

        return masked




if __name__ == "__main__":
    img = Image('images/dog.jpg')
    print(img.img.shape)
    print(img.gray_img.shape)
