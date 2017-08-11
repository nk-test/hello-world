import matplotlib.pyplot as plt
from matplotlib.image import imread

if __name__ == '__main__':
    img = imread('deep-learning-from-scratch-master/dataset/lena.png')
    plt.imshow(img)
    plt.show()
