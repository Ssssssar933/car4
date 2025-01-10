import cv2
import numpy as np

def calculate_slope(line):
        """
    计算线段line的斜率
    :param line: np.array([[x_1, y_1, x_2, y_2]])
    :return:
    """
        x_1, y_1, x_2, y_2 = line[0]
        if(x_2 - x_1==0):
            return float('inf')#更改垂直
        return (y_2 - y_1)/(x_2 - x_1)
def reject_abnormal_lines(lines, threshold):
    """
    剔除斜率不一致的线段
    :param lines:线段合集,[np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]])]
    """
    #列表生成式，遍历所有的lines对每条lines求取斜率得到总斜率
    slopes = [calculate_slope(line) for line in lines if calculate_slope(line) != float('inf')]
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
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
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

def show_lane(frame):
    """
    在frame上画出车道线
    :param frame: 视频帧,是一个NumPy数组
    :return:
    """
    # 将彩色帧转换为灰度图像
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 应用边缘检测
    edge_img = cv2.Canny(gray_img, 100, 150)
    # 应用掩码
    mask = np.zeros_like(edge_img)
    mask = cv2.fillPoly(mask, np.array([[[0,900],[550,550],[860,550],[1200,900]]]), color=255)
    masked_edge_img = cv2.bitwise_and(edge_img, mask)
    
    # 通过Hough变换获取所有线段
    lines = cv2.HoughLinesP(masked_edge_img, 1, np.pi / 180, 15, minLineLength=40, maxLineGap=20)
    
    # 检查是否成功检测到线段
    if lines is not None:
        # 按照斜率分成车道线
        left_lines = [line for line in lines if calculate_slope(line) > 0]
        right_lines = [line for line in lines if calculate_slope(line) < 0]
        
        # 剔除斜率异常的线段
        left_lines = reject_abnormal_lines(left_lines, threshold=0.2)
        right_lines = reject_abnormal_lines(right_lines, threshold=0.2)
        
        print("左车道线坐标")
        print(least_squares_fit(left_lines))
        print("右车道线坐标")
        print(least_squares_fit(right_lines))

        # 最小二乘拟合，得到车道线的两个端点
        left_line = least_squares_fit(left_lines)
        right_line = least_squares_fit(right_lines)
        
        # 在原彩色帧上绘制车道线
        if left_line is not None:
            cv2.line(frame, tuple(left_line[0]), tuple(left_line[1]), color=(0, 255, 255), thickness=3)
        if right_line is not None:
            cv2.line(frame, tuple(right_line[0]), tuple(right_line[1]), color=(0, 255, 255), thickness=3)
    else:
        # 如果没有检测到线段，可以输出提示信息或者采取其他措施
        print("未能检测到线段，请检查输入数据或算法参数。")
    
    return frame

capture = cv2.VideoCapture('video6.mp4')
while True:
    ret,frame = capture.read()
    frame = show_lane(frame)
    cv2.imshow('frame',frame)
    cv2.waitKey(25)