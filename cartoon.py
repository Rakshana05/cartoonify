import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Image
def read_file(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    return img

# Create edge mask
def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges

# Reduce the colours
def color_quantization(img, k):
    data = np.float32(img).reshape((-1,3))
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

# Combine edge masking and quantising
def cartoon(blurred):
    c= cv2.bitwise_and(blurred, blurred, mask= edges)
    plt.imshow(c)
    plt.show()

img = read_file("Elon.jpeg")
line_size, blur_value = 3,7 ## Why only this works??
edges = edge_mask(img, line_size, blur_value)
img = color_quantization(img, 9) # do trial and error to find a good k value

# Reduce Noise
blurred = cv2.bilateralFilter(img, d=5, sigmaColor=100, sigmaSpace=100)

plt.imshow(blurred)
plt.show()

# plt.imshow(img)
# plt.show()
cartoon(blurred)

# plt.imshow(edges, cmap='binary')
# plt.show()


''' for Elon -> just the blurred image is good, 
but for Virat -> it doesnt have much difference even when adding outline
LineSize - outline thickness!:)
try out with odd numbers'''
