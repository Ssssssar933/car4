import cv2
import numpy as np

edge_img = cv2.imread('edges_img.jpg', cv2.IMREAD_GRAYSCALE)

mask = np.zeros_like(edge_img)
mask = cv2.fillPoly(mask, np.array([[[0,900],[550,550],[860,550],[1200,900]]]),color=255)
#cv2.imshow('mask', mask)
#cv2.waitKey(0)

#掩码之后的图像
masked_edge_img = cv2.bitwise_and(edge_img, mask)
cv2.imshow('maskes', masked_edge_img)
cv2.waitKey(0)
cv2.imwrite('masked_edge_img.jpg',masked_edge_img)