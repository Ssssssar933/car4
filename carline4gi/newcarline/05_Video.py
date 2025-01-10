import cv2
import numpy as np

def get_edge_img(color_img):
    #灰度模式读取图像
    img = cv2.imread(color_img,cv2.IMREAD_GRAYSCALE) 
    #打印格式
    print(type(img))
    #打印矩阵形状     
    print(img.shape)
    #展示图像
    #cv2.imshow('image',img)   
    #cv2.waitKey(0)  防止图像闪退让程序阻塞
    #将内存中的图片写入文件
    cv2.imwrite('img_gray.jpg',img)

def roi_mask(edge_img):
    img = cv2.imread(edge_img,cv2.IMREAD_GRAYSCALE)
    #得到一个保存了边缘值的矩阵,可调整阈值来减少弱边缘
    edge_img = cv2.Canny(img, 200, 200)
    #cv2.imshow('edges',edge_img)
    cv2.waitKey(0)
    cv2.imwrite('edges_img.jpg',edge_img)

def get_lines(mask_gray_img):
    edge_img = cv2.imread(mask_gray_img, cv2.IMREAD_GRAYSCALE)
    mask = np.zeros_like(edge_img)
    mask = cv2.fillPoly(mask, np.array([[[0,320],[250,200],[270,200],[540,320]]]),color=255)
    #cv2.imshow('mask', mask)
    #cv2.waitKey(0)
    #掩码之后的图像
    masked_edge_img = cv2.bitwise_and(edge_img, mask)
    #cv2.imshow('maskes', masked_edge_img)
    cv2.waitKey(0)
    cv2.imwrite('masked_edge_img.jpg',masked_edge_img)

def calculate_slope(line):
        """
    计算线段line的斜率
    :param line: np.array([[x_1, y_1, x_2, y_2]])
    :return:
    """
        x_1, y_1, x_2, y_2 = line[0]
        return (y_2 - y_1)/(x_2 - x_1)
def reject_abnormal_lines(lines, threshold):
    """
    剔除斜率不一致的线段
    :param lines:线段合集,[np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]])]
    """
    #列表生成式，遍历所有的lines对每条lines求取斜率得到总斜率
    slopes = [calculate_slope(line) for line in lines]
    while len(lines) > 0:
        mean = np.mean(slopes)    #计算所有斜率的平均值 
        diff = [abs(s-mean) for s in slopes]    #计算每一条斜率与平均斜率的差值
        idx = np.argmax(diff)      #找到差值最大的直线下标
        if diff[idx] > threshold:
            slopes.pop(idx)  #重新计算斜率
            lines.pop(idx)   #在集合中删除这条直线
        else:
            break
    return lines
def least_squares_fit(lines):
    """
    最小二乘拟合,将lines中的线段拟合成一条线段
    :param lines:线段集合,[np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
    :return: 线段上的两点,np.array([[xmin,ymin],[xmax,ymax]])
    """
    #取出所有坐标点
    x_coords = []
    y_coords = []
    for line in lines:
        if len(line.shape) == 2 and line.shape[0] == 1:
            x_1, y_1, x_2, y_2 = line[0]
            x_coords.extend([x_1, x_2])
            y_coords.extend([y_1, y_2])
        elif len(line.shape) == 1 and line.shape[0] == 4:
            x_1, y_1, x_2, y_2 = line
            x_coords.extend([x_1, x_2])
            y_coords.extend([y_1, y_2])
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    #进行直线拟合，得到多项式系数
    poly = np.polyfit(x_coords, y_coords, deg=1)
    #根据多项式系数，计算两个直线上的点，用于唯一确定这条直线
    point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
    return np.array([point_min, point_max], dtype=np.int64)
    
def draw_lines(color_img,lines):
    edge_img = cv2.imread(color_img,cv2.IMREAD_GRAYSCALE)
    #通过Hough变换获取所有线段
    lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, 15, minLineLength=40, maxLineGap=20)

    #按照斜率分成车道线
    left_lines = [line for line in lines if calculate_slope(line) > 0]
    right_lines = [line for line in lines if calculate_slope(line) < 0]
    print('剔除离群值之前')
    print('左车道线数量')
    print(len(left_lines))
    print('右车道线数量')
    print(len(right_lines))

    left_lines = reject_abnormal_lines(left_lines, threshold=0.2)
    right_lines = reject_abnormal_lines(right_lines, threshold=0.2)

    print('剔除离群值之后')
    print('左车道线数量')
    print(len(left_lines))
    print('右车道线数量')
    print(len(right_lines))
    print("左车道线坐标")
    print(least_squares_fit(left_lines))
    print("右车道线坐标")
    print(least_squares_fit(right_lines))

    left_line = least_squares_fit(left_lines)
    right_line = least_squares_fit(right_lines)

    img = cv2.imread(color_img,cv2.IMREAD_COLOR)
    cv2.line(img, tuple(left_line[0]), tuple(left_line[1]), color=(0, 255, 255), thickness=3)
    cv2.line(img, tuple(right_line[0]), tuple(right_line[1]), color=(0, 255, 255), thickness=3)

    cv2.imshow('lane', img)
    cv2.waitKey(0)

def show_lane(color_img):
    """
        在color_img上画出车道线
        :param color_img:彩色图,channels=3
        :return:
        """
    edge_img = get_edge_img(color_img)
    mask_gray_img = roi_mask(edge_img)
    lines = get_lines(mask_gray_img)
    draw_lines(color_img,lines)
    return color_img

capture = cv2.VideoCapture('video.mp4')
while True:
    ret,frame = capture.read()
    frame = show_lane(frame)
    cv2.imshow('frame',frame)
    cv2.waitKey(25)