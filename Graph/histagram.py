import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "Times New Roman"
img_size = 224
# img = np.zeros((500,500), np.uint8)
img = cv.imread("Leafspot.jpeg")
img2 = cv.resize(img, (img_size, img_size))
cv.imshow("img", img2)
plt.hist(img2.ravel(), 256, [0, 256])
plt.xlabel('pixel')
plt.ylabel("intensity")
plt.savefig('LeafHista.jpg')
plt.show()

img = cv.imread("Sound.jpg")
img2 = cv.resize(img, (img_size, img_size))
cv.imshow("img", img2)
plt.hist(img2.ravel(), 256, [0, 256])
plt.xlabel('pixel')
plt.ylabel("intensity")
plt.savefig('SoundHista.jpg')
plt.show()


img = cv.imread("Rust.jpeg")
img2 = cv.resize(img, (img_size, img_size))
cv.imshow("img", img2)
plt.hist(img2.ravel(), 256, [0, 256])
plt.xlabel('pixel')
plt.ylabel("intensity")
plt.savefig('RustHista.jpg')
plt.show()


img = cv.imread("whitefly.jpeg")
img2 = cv.resize(img, (img_size, img_size))
cv.imshow("img", img2)
plt.hist(img2.ravel(), 256, [0, 256])
plt.xlabel('pixel')
plt.ylabel("intensity")
plt.savefig('whiteflyHista.jpg')
plt.show()
