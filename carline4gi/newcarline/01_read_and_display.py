import cv2
 
#灰度模式读取图像
img = cv2.imread('carline.jpg',cv2.IMREAD_GRAYSCALE) 
#打印格式
print(type(img))
#打印矩阵形状     
print(img.shape)

#展示图像
cv2.imshow('image',img)   
#cv2.waitKey(0)  防止图像闪退让程序阻塞

#将内存中的图片写入文件
cv2.imwrite('img_gray.jpg',img)