from PyQt5 import Qt
from PyQt5 import QtCore,QtWidgets,QtGui
import sys
import PyQt5
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QFileDialog, QGraphicsRectItem, QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QSize
import cv2
import numpy as np
from matplotlib import pyplot as plt

import window
import window2


class MainWindow():
    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        self.raw_image = None
        self.ui = window.Ui_MainWindow()
        self.ui.setupUi(MainWindow)
        self.action_connect()
        MainWindow.show()
        sys.exit(app.exec_())


# 信号槽绑定
    def action_connect(self):
        self.ui.action.triggered.connect(self.open_file)
        self.ui.action_2.triggered.connect(self.save_file)
        self.ui.action_5.triggered.connect(self.recover_img)
        # 饱和度
        self.ui.horizontalSlider.valueChanged.connect(self.slider_change)
        self.ui.horizontalSlider.sliderReleased.connect(self.show_histogram)

        # 亮度
        self.ui.horizontalSlider_4.valueChanged.connect(self.slider_change)
        self.ui.horizontalSlider_4.sliderReleased.connect(self.show_histogram)

        # 美白（人脸识别）
        self.ui.horizontalSlider_8.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_8.sliderReleased.connect(self.show_histogram)

        # 美白（皮肤识别）
        self.ui.horizontalSlider_13.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_13.sliderReleased.connect(self.show_histogram)

        # 磨皮精度
        self.ui.horizontalSlider_14.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_14.sliderReleased.connect(self.show_histogram)

        # 磨皮程度
        self.ui.horizontalSlider_11.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_11.sliderReleased.connect(self.show_histogram)

        # 伽马变换
        self.ui.horizontalSlider_5.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_5.sliderReleased.connect(self.show_histogram)

        # 人脸识别和皮肤检测
        self.ui.tabWidget.tabBarClicked.connect(self.calculate)

        # 木刻滤镜
        self.ui.horizontalSlider_9.sliderReleased.connect(self.woodcut)
        self.ui.horizontalSlider_9.sliderReleased.connect(self.show_histogram)

        # 灰色铅笔画
        self.ui.horizontalSlider_7.sliderReleased.connect(self.pencil_gray)
        self.ui.horizontalSlider_7.sliderReleased.connect(self.show_histogram)

        # 怀旧滤镜
        self.ui.horizontalSlider_10.sliderReleased.connect(self.reminiscene)
        self.ui.horizontalSlider_10.sliderReleased.connect(self.show_histogram)

        # 铅笔画滤镜
        self.ui.horizontalSlider_12.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_12.sliderReleased.connect(self.show_histogram)

        # 风格化
        self.ui.horizontalSlider_2.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_2.sliderReleased.connect(self.show_histogram)

        # 细节增强
        self.ui.horizontalSlider_6.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_6.sliderReleased.connect(self.show_histogram)

        # 边缘保持
        self.ui.horizontalSlider_3.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_3.sliderReleased.connect(self.show_histogram)

        # 打开摄像头
        self.ui.action_17.triggered.connect(self.new_camera)

        # 标记人脸位置
        self.ui.action_18.triggered.connect(self.mark_face)

# 显示图片
    def show_image(self):
        img_cv = cv2.cvtColor(self.current_img, cv2.COLOR_RGB2BGR)
        img_width, img_height, a = img_cv.shape
        ratio_img = img_width/img_height
        ratio_scene = self.ui.graphicsView.width()/self.ui.graphicsView.height()
        if ratio_img > ratio_scene:
            width = int(self.ui.graphicsView.width())
            height = int(self.ui.graphicsView.width() / ratio_img)
        else:
            width = int(self.ui.graphicsView.height() * ratio_img)
            height = int(self.ui.graphicsView.height())
        img_resize = cv2.resize(img_cv, (height-5, width-5), interpolation=cv2.INTER_AREA)
        h, w, c = img_resize.shape
        bytesPerLine = w * 3
        qimg = QImage(img_resize.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.scene = QGraphicsScene()
        pix = QPixmap(qimg)
        self.scene.addPixmap(pix)
        self.ui.graphicsView.setScene(self.scene)

# 显示灰度图像
    def show_grayimage(self):
        img_cv = self.gray_image
        img_width, img_height = img_cv.shape
        ratio_img = img_width/img_height
        ratio_scene = self.ui.graphicsView.width()/self.ui.graphicsView.height()
        if ratio_img > ratio_scene:
            width = int(self.ui.graphicsView.width())
            height = int(self.ui.graphicsView.width() / ratio_img)
        else:
            width = int(self.ui.graphicsView.height() * ratio_img)
            height = int(self.ui.graphicsView.height())
        img_resize = cv2.resize(img_cv, (height-5, width-5), interpolation=cv2.INTER_AREA)
        h, w = img_resize.shape
        qimg = QImage(img_resize.data, w, h, w, QImage.Format_Grayscale8)
        self.scene = QGraphicsScene()
        pix = QPixmap(qimg)
        self.scene.addPixmap(pix)
        self.ui.graphicsView.setScene(self.scene)


# 显示直方图
    def show_histogram(self):
        if self.raw_image is None:
            return 0
        img = self.current_img
        plt.figure(figsize=((self.ui.tab_3.width()-10)/100, (self.ui.tab_3.width()-60)/100), frameon=False)
        plt.hist(img.ravel(), bins=256, range=[0, 256])
        plt.axes().get_yaxis().set_visible(False)
        # plt.axes().get_xaxis().set_visible(False)
        ax = plt.axes()
        # 隐藏坐标系的外围框线
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.savefig('Hist.png', bbox_inches="tight", transparent=True, dpi=100)
        pix = QPixmap("Hist.png")
        self.ui.label.setPixmap(pix)
        self.ui.label_2.setPixmap(pix)
        self.ui.label_3.setPixmap(pix)

# 保存图片
    def save_file(self):
        fname = QFileDialog.getSaveFileName(None, '打开文件', './', ("Images (*.png *.xpm *.jpg)"))
        if fname[0]:
            cv2.imwrite(fname[0], self.current_img)

# 打开图片
    def open_file(self):
        fname = QFileDialog.getOpenFileName(None, '打开文件', './', ("Images (*.png *.xpm *.jpg)"))
        if fname[0]:
            img_cv = cv2.imdecode(np.fromfile(fname[0], dtype=np.uint8), -1)  # 注意这里读取的是RGB空间的
            self.raw_image = img_cv
            self.last_image = img_cv
            self.current_img = img_cv
            self.show_image()
            self.show_histogram()
            self.imgskin = np.zeros(self.raw_image.shape)
        self.intial_value()

# 恢复图片
    def recover_img(self):
        self.current_img = self.raw_image
        self.show_image()
        self.show_histogram()
        self.intial_value()

# 饱和度
    def change_saturation(self):
        if self.raw_image is None:
            return 0

        value = self.ui.horizontalSlider.value()
        img_hsv = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2HLS)
        if value > 2:
            img_hsv[:, :, 2] = np.log(img_hsv[:, :, 2] /255* (value - 1)+1) / np.log(value + 1) * 255
        if value < 0:
            img_hsv[:, :, 2] = np.uint8(img_hsv[:, :, 2] / np.log(- value + np.e))
        self.current_img = cv2.cvtColor(img_hsv, cv2.COLOR_HLS2BGR)

# 明度调节
    def change_darker(self):
        if self.raw_image is None:
            return 0
        value = self.ui.horizontalSlider_4.value()
        img_hsv = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2HLS)
        if value > 3:
            img_hsv[:, :, 1] = np.log(img_hsv[:, :, 1] /255* (value - 1)+1) / np.log(value + 1) * 255
        if value < 0:
            img_hsv[:, :, 1] = np.uint8(img_hsv[:, :, 1] / np.log(- value + np.e))
        self.last_image = self.current_img
        self.current_img = cv2.cvtColor(img_hsv, cv2.COLOR_HLS2BGR)

# 人脸识别
    def detect_face(self):
        img = self.raw_image
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces

# 皮肤识别
    def detect_skin(self):
        img = self.raw_image
        rows, cols, channals = img.shape
        for r in range(rows):
            for c in range(cols):
                B = img.item(r, c, 0)
                G = img.item(r, c, 1)
                R = img.item(r, c, 2)
                if (abs(R - G) > 15) and (R > G) and (R > B):
                    if (R > 95) and (G > 40) and (B > 20) and (max(R, G, B) - min(R, G, B) > 15):
                        self.imgskin[r, c] = (1, 1, 1)
                    elif (R > 220) and (G > 210) and (B > 170):
                        self.imgskin[r, c] = (1, 1, 1)

# 皮肤磨皮(value1精细度，value2程度)
    def dermabrasion(self, value1=3, value2=2):
        value1 = self.ui.horizontalSlider_14.value()
        value2 = 11 - self.ui.horizontalSlider_11.value()
        if value1 == 0 and value2 == 0:
            return 0
        if value2 == 0:
            value2 = 2
        if value1 == 0:
            value1 = 3
        img = self.current_img
        dx = value1 * 5
        fc = value1 * 12.5
        p = 50
        temp1 = cv2.bilateralFilter(img, dx, fc, fc)
        temp2 = (temp1 - img + 128)
        temp3 = cv2.GaussianBlur(temp2, (2 * value2 - 1, 2 * value2 - 1), 0, 0)
        temp4 = img + 2 * temp3 - 255
        dst = np.uint8(img * ((100 - p) / 100) + temp4 * (p / 100))


        imgskin_c = np.uint8(-(self.imgskin - 1))

        dst = np.uint8(dst * self.imgskin + img * imgskin_c)
        self.current_img = dst

# 美白算法(皮肤识别)
    def whitening_skin(self, value=30):
        # value = 30
        value = self.ui.horizontalSlider_13.value()
        img = self.current_img
        imgw = np.zeros(img.shape, dtype='uint8')
        imgw = img.copy()
        midtones_add = np.zeros(256)

        for i in range(256):
            midtones_add[i] = 0.667 * (1 - ((i - 127.0) / 127) * ((i - 127.0) / 127))

        lookup = np.zeros(256, dtype="uint8")

        for i in range(256):
            red = i
            red += np.uint8(value * midtones_add[red])
            red = max(0, min(0xff, red))
            lookup[i] = np.uint8(red)



        rows, cols, channals = img.shape
        for r in range(rows):
            for c in range(cols):

                if self.imgskin[r, c, 0] == 1:
                    imgw[r, c, 0] = lookup[imgw[r, c, 0]]
                    imgw[r, c, 1] = lookup[imgw[r, c, 1]]
                    imgw[r, c, 2] = lookup[imgw[r, c, 2]]
        self.current_img = imgw

# 美白算法(人脸识别)
    def whitening_face(self, value=30):
        # value = 30
        value = self.ui.horizontalSlider_8.value()
        img = self.current_img
        imgw = np.zeros(img.shape, dtype='uint8')
        imgw = img.copy()
        midtones_add = np.zeros(256)

        for i in range(256):
            midtones_add[i] = 0.667 * (1 - ((i - 127.0) / 127) * ((i - 127.0) / 127))

        lookup = np.zeros(256, dtype="uint8")

        for i in range(256):
            red = i
            red += np.uint8(value * midtones_add[red])
            red = max(0, min(0xff, red))
            lookup[i] = np.uint8(red)

        # faces可全局变量
        faces = self.faces

        if faces == ():
            rows, cols, channals = img.shape
            for r in range(rows):
                for c in range(cols):
                    imgw[r, c, 0] = lookup[imgw[r, c, 0]]
                    imgw[r, c, 1] = lookup[imgw[r, c, 1]]
                    imgw[r, c, 2] = lookup[imgw[r, c, 2]]

        else:
            x, y, w, h = faces[0]
            rows, cols, channals = img.shape
            x = max(x - (w * np.sqrt(2) - w) / 2, 0)
            y = max(y - (h * np.sqrt(2) - h) / 2, 0)
            w = w * np.sqrt(2)
            h = h * np.sqrt(2)
            rows = min(rows, y + h)
            cols = min(cols, x + w)
            for r in range(int(y), int(rows)):
                for c in range(int(x), int(cols)):
                    imgw[r, c, 0] = lookup[imgw[r, c, 0]]
                    imgw[r, c, 1] = lookup[imgw[r, c, 1]]
                    imgw[r, c, 2] = lookup[imgw[r, c, 2]]

            processWidth = int(max(min(rows - y, cols - 1) / 8, 2))
            for i in range(1, processWidth):
                alpha = (i - 1) / processWidth
                for r in range(int(y), int(rows)):
                    imgw[r, int(x) + i - 1] = np.uint8(
                        imgw[r, int(x) + i - 1] * alpha + img[r, int(x) + i - 1] * (1 - alpha))
                    imgw[r, int(cols) - i] = np.uint8(
                        imgw[r, int(cols) - i] * alpha + img[r, int(cols) - i] * (1 - alpha))
                for c in range(int(x) + processWidth, int(cols) - processWidth):
                    imgw[int(y) + i - 1, c] = np.uint8(
                        imgw[int(y) + i - 1, c] * alpha + img[int(y) + i - 1, c] * (1 - alpha))
                    imgw[int(rows) - i, c] = np.uint8(
                        imgw[int(rows) - i, c] * alpha + img[int(rows) - i, c] * (1 - alpha))
        self.current_img = imgw

# Gamma矫正
    def gamma_trans(self):
        gamma = (self.ui.horizontalSlider_5.value() + 10) / 10
        img = self.current_img
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        self.current_img = cv2.LUT(img, gamma_table)
        self.show_image()
        self.show_histogram()

# 响应滑动条的变化
    def slider_change(self):
        if self.raw_image is None:
            return 0

        self.current_img = self.raw_image

        # 伽马变换
        if self.ui.horizontalSlider_5.value() != 0:
            self.gamma_trans()

        # 饱和度
        if self.ui.horizontalSlider.value() != 0:
            self.change_saturation()

        if self.ui.horizontalSlider_2.value() != 0:
            pass

        # 边缘保持
        if self.ui.horizontalSlider_3.value() != 0:
            self.edge_preserve()

        # 亮度
        if self.ui.horizontalSlider_4.value() != 0:
            self.change_darker()

        # 美白（人脸识别）
        if self.ui.horizontalSlider_8.value() != 0:
            self.whitening_face()

        # 美白（皮肤识别）
        if self.ui.horizontalSlider_13.value() != 0:
            self.whitening_skin()

        # 磨皮程度
        if self.ui.horizontalSlider_11.value() != 0:
                self.dermabrasion()

        # 磨皮精度
        if self.ui.horizontalSlider_14.value() != 0:
            self.dermabrasion()

        # 风格化
        if self.ui.horizontalSlider_2.value() != 0:
            self.stylize()

        # 细节增强
        if self.ui.horizontalSlider_6.value() != 0:
            self.detail_enhance()

        # 铅笔画
        if self.ui.horizontalSlider_12.value() != 0:
            self.pencil_color()
        self.show_image()


# 计算人脸识别和皮肤识别的基本参数
    def calculate(self):
        if self.raw_image is None:
            return 0
        if self.calculated is False:
            self.faces = self.detect_face()
            if self.faces != ():
                self.detect_skin()
            self.calculated = True

# 怀旧滤镜
    def reminiscene(self):
        if self.raw_image is None:
            return 0
        if self.ui.horizontalSlider_10.value() == 0:
            self.current_img = self.raw_image
            self.show_image()
            return 0
        img = self.raw_image.copy()
        rows, cols, channals = img.shape
        for r in range(rows):
            for c in range(cols):
                B = img.item(r, c, 0)
                G = img.item(r, c, 1)
                R = img.item(r, c, 2)
                img[r, c, 0] = np.uint8(min(max(0.272 * R + 0.534 * G + 0.131 * B, 0), 255))
                img[r, c, 1] = np.uint8(min(max(0.349 * R + 0.686 * G + 0.168 * B, 0), 255))
                img[r, c, 2] = np.uint8(min(max(0.393 * R + 0.769 * G + 0.189 * B, 0), 255))
        self.current_img = img
        self.show_image()

# 木刻滤镜
    def woodcut(self):
        if self.raw_image is None:
            return 0
        if self.ui.horizontalSlider_9.value() == 0:
            # self.current_img = self.raw_image
            self.show_image()
            return 0
        self.gray_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        gray = self.gray_image
        value = 70 + self.ui.horizontalSlider_9.value()
        rows, cols = gray.shape
        for r in range(rows):
            for c in range(cols):
                if gray[r, c] > value:
                    gray[r, c] = 255
                else:
                    gray[r, c] = 0
        self.gray_image = gray
        self.show_grayimage()

# 铅笔画(灰度)
    def pencil_gray(self):
        if self.raw_image is None:
            return 0
        if self.ui.horizontalSlider_7.value() == 0:
            # self.current_img = self.raw_image
            self.show_image()
            return 0
        value = self.ui.horizontalSlider_7.value() * 0.05
        dst1_gray, dst1_color = cv2.pencilSketch(self.current_img, sigma_s=50, sigma_r=value, shade_factor=0.04)
        self.gray_image = dst1_gray
        self.show_grayimage()

# 铅笔画(彩色)
    def pencil_color(self):
        if self.raw_image is None:
            return 0
        if self.ui.horizontalSlider_12.value() == 0:
            self.current_img = self.raw_image
            self.show_image()
            return 0
        value = self.ui.horizontalSlider_12.value() * 0.05
        dst1_gray, dst1_color = cv2.pencilSketch(self.current_img, sigma_s=50, sigma_r=value, shade_factor=0.04)
        self.current_img = dst1_color


# 风格化
    def stylize(self):
        if self.raw_image is None:
            return 0
        if self.ui.horizontalSlider_2.value() == 0:
            self.current_img = self.raw_image
            self.show_image()
            return 0
        value = self.ui.horizontalSlider_2.value() * 0.05
        self.current_img = cv2.stylization(self.current_img, sigma_s=50, sigma_r=value)


# 细节增强
    def detail_enhance(self):
        if self.raw_image is None:
            return 0
        if self.ui.horizontalSlider_6.value() == 0:
            self.current_img = self.raw_image
            self.show_image()
            return 0
        value = self.ui.horizontalSlider_6.value() * 0.05
        self.current_img = cv2.detailEnhance(self.current_img, sigma_s=50, sigma_r=value)

# 边缘保持
    def edge_preserve(self):
        if self.raw_image is None:
            return 0
        if self.ui.horizontalSlider_3.value() == 0:
            self.current_img = self.raw_image
            self.show_image()
            return 0
        value = self.ui.horizontalSlider_3.value() * 0.05
        self.current_img = cv2.edgePreservingFilter(self.current_img, flags=1, sigma_s=50, sigma_r=value)

# 显示摄像照片
    def show_camera(self):
        flag, self.camera_image = self.cap.read()
        show = cv2.resize(self.image, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

# 初始化
    def intial_value(self):
        self.calculated = False
        self.ui.horizontalSlider.setValue(0)
        self.ui.horizontalSlider_2.setValue(0)
        self.ui.horizontalSlider_3.setValue(0)
        self.ui.horizontalSlider_4.setValue(0)
        self.ui.horizontalSlider_5.setValue(0)
        self.ui.horizontalSlider_6.setValue(0)
        self.ui.horizontalSlider_7.setValue(0)
        self.ui.horizontalSlider_8.setValue(0)
        self.ui.horizontalSlider_9.setValue(0)
        self.ui.horizontalSlider_10.setValue(0)
        self.ui.horizontalSlider_11.setValue(0)
        self.ui.horizontalSlider_12.setValue(0)
        self.ui.horizontalSlider_13.setValue(0)
        self.ui.horizontalSlider_14.setValue(0)

# 调用摄像头窗口
    def new_camera(self):
        Dialog = QtWidgets.QDialog()
        self.ui_2 = window2.Ui_Form()
        self.ui_2.setupUi(Dialog)
        Dialog.show()
        self.ui_2.pushButton_2.clicked.connect(self.get_image)
        Dialog.exec_()
        if self.ui_2.cap.isOpened():
            self.ui_2.cap.release()
        if self.ui_2.timer_camera.isActive():
            self.ui_2.timer_camera.stop()

# 获取摄像头的图片
    def get_image(self):
        if self.ui_2.captured_image is not None:
            self.raw_image = self.ui_2.captured_image
            self.current_img = self.ui_2.captured_image
            self.show_image()
            self.show_histogram()
            self.imgskin = np.zeros(self.raw_image.shape)
            self.intial_value()

# 显示人脸识别
    def mark_face(self):
        if self.raw_image is None:
            return 0
        if self.calculated == False:
            self.calculate()
        for (x, y, w, h) in self.faces:
            self.current_img = cv2.rectangle(self.current_img.copy(), (x, y), (x+w, y+h), (255, 0, 0), 1)
        self.show_image()


if __name__ == "__main__":
    MainWindow()
