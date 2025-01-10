import cv2

img = cv2.imread('carline.jpg',cv2.IMREAD_GRAYSCALE)

#得到一个保存了边缘值的矩阵,可调整阈值来减少弱边缘
edge_img = cv2.Canny(img, 100, 150)

cv2.imshow('edges',edge_img)
cv2.waitKey(0)

cv2.imwrite('edges_img.jpg',edge_img)
