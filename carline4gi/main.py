from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from plyer import filechooser
import cv2
import numpy as np
import time

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
        if not diff:
            return lines
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
    if lines is None:
        print("传入线段集合为None")
        return
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

    #cv2.imshow('lane', img)
    #cv2.waitKey(0)
    return img
def show_lane(frame):
    """
    在frame上画出车道线
    :param frame: 视频帧,是一个NumPy数组
    :return:
    """
    # 将彩色帧转换为灰度图像
    gray_img = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
    # 应用高斯模糊
    #blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # 应用边缘检测
    edge_img = cv2.Canny(gray_img, 100, 200)
    #cv2.imshow('edge', edge_img)
    # 应用掩码
    mask = np.zeros_like(edge_img)
    mask = cv2.fillPoly(mask, np.array([[[0,700],[300,400],[900,400],[1300,700]]]), color=255)
    masked_edge_img = cv2.bitwise_and(edge_img, mask)
    #cv2.imshow('maskes', masked_edge_img)
    
    # 通过Hough变换获取所有线段
    lines = cv2.HoughLinesP(masked_edge_img, 1, np.pi / 180, 15, minLineLength=40, maxLineGap=20)
    
    # 检查是否成功检测到线段
    if lines is not None:
        if lines is None:
            print("传入线段集合为None")
            return
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

        if left_line is not None:
            cv2.line(frame, tuple(left_line[0]), tuple(left_line[1]), color=(0, 255, 255), thickness=3)
        if right_line is not None:
            cv2.line(frame, tuple(right_line[0]), tuple(right_line[1]), color=(0, 255, 255), thickness=3)
    else:
        # 如果没有检测到线段，可以输出提示信息或者采取其他措施
        print("未能检测到线段，请检查输入数据或算法参数。")
    
    return frame

class MyApp(App):
    def build(self):
        Window.size = (720, 1280)
        layout = BoxLayout(orientation='vertical',size_hint=(None,None),width=720,height=1280)

        # 打开摄像头按钮
        open_camera_btn = Button(text='open Camera')
        open_camera_btn.bind(on_press=self.open_camera)

        # 浏览本地文件按钮
        browse_file_btn = Button(text='browse file')
        browse_file_btn.bind(on_press=self.browse_file)

        layout.add_widget(open_camera_btn)
        layout.add_widget(browse_file_btn)

        return layout

    def open_camera(self, instance):
        self.cap = cv2.VideoCapture(0)
        while True:
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow('Camera', frame)
                processed_frame = self.process_image(frame)  # 处理图像
                self.display_image(processed_frame)  # 显示处理后的图像
                if cv2.waitKey(1) & 0xFF == 27:  # 按ESC键退出
                    break
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def browse_file(self, instance):
        filechooser.open_file(on_selection=self.handle_file_selection)

    def handle_file_selection(self, selection):
        if selection:
            self.cap = cv2.VideoCapture(selection[0])
            if not self.cap.isOpened():
                print("failed to open file.")
                return
            Clock.schedule_interval(self.process_video_frame, 1 / 25)
        else:
            print("No file select")        
        

    def process_video_frame(self,dt):
        start_frame_time = time.perf_counter()  # 记录每帧处理开始时间
        ret,frame = self.cap.read()
        if ret:
            processed_image = self.process_image(frame)  # 处理图像
            self.display_image(processed_image)  # 显示处理后的图像
            end_frame_time = time.perf_counter()  # 记录每帧处理结束时间
            print(f"单帧处理耗时: {end_frame_time - start_frame_time} 秒")
        else:
            print("视频读取结束")
            self.cap.release()
            cv2.destroyAllWindows()
            Clock.unschedule(self.process_video_frame)

    def process_image(self, frame):
        """
        图像处理算法
        """
        # 将图像进行旋转纠正方向
        #frame = cv2.transpose(frame)
        #frame = cv2.flip(frame, 0)
        processed_frame = show_lane(frame)
        return processed_frame

    def display_image(self, image):
        """
        将处理后的图像显示在Kivy界面上
        """
        image = cv2.flip(image, 0)
        texture = Texture.create(size=(image.shape[1], image.shape[0]))
        texture.blit_buffer(image.tobytes(order='C'), colorfmt='bgr', bufferfmt='ubyte')
        if not hasattr(self, 'img_widget'):
        # 如果还不存在img_widget属性，说明是首次调用该方法，创建新的图像部件并添加到界面
            self.img_widget = Image(texture=texture)
            self.root.add_widget(self.img_widget)
        else:
            # 如果img_widget已经存在，直接更新其纹理数据来显示新的图像内容
            self.img_widget.texture = texture
        return self.img_widget


if __name__ == '__main__':
    MyApp().run()