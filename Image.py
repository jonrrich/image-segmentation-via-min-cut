import matplotlib.pyplot as plt

class Image:
    def __init__(self, path):
        self.img = plt.imread(path)
        self.gray_img = 0.2989*self.img[:,:,0] + 0.5870*self.img[:,:,1] + 0.1140*self.img[:,:,2]


if __name__ == "__main__":
    img = Image('images/dog.jpg')
    print(img.img.shape)
    print(img.gray_img.shape)
