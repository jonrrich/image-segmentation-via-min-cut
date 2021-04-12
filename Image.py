import matplotlib.pyplot as plt

class Image:
    def __init__(self, path):
        self.img = plt.imread(path)
        self.gray_img = 0.2989*self.img[:,:,0] + 0.5870*self.img[:,:,1] + 0.1140*self.img[:,:,2]


    def segmentation(self,obj_pixels):
        segmented = np.zeros_like(self.img)
        obj_pixels = np.delete(obj_pixels, np.where(arr == "S"))

        for pixel in obj_pixels:
            segmented[pixel] = self.img[pixel]

        return segmented



if __name__ == "__main__":
    img = Image('images/dog.jpg')
    print(img.img.shape)
    print(img.gray_img.shape)
