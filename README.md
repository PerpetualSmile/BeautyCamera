# BeautyCamera（PyQt5+python-opencv+matplotlib）
![](https://raw.githubusercontent.com/PerpetualSmile/picture/master/BeautyCamera/BeautyCamera.png)

![](https://raw.githubusercontent.com/PerpetualSmile/picture/master/BeautyCamera/BeautyCamera_1.png)

![](https://raw.githubusercontent.com/PerpetualSmile/picture/master/BeautyCamera/BeautyCamera_2.png)

![](https://raw.githubusercontent.com/PerpetualSmile/picture/master/BeautyCamera/BeautyCamera_3.png)


## 打开图片
- 美颜相机demo既可以处理已有的图片，也可以用本程序打开电脑的摄像头进行拍摄，相关选项在文件菜单下有，也可以使用快捷键crtl+o打开图片。
## 拍摄照片
- 在文件菜单下有打开摄像头选项，点击可以跳出摄像窗口，再点击弹窗中的打开摄像头按钮可以开启摄像头，选好景之后点击拍摄按钮可以将图片显示在主窗口进行各种处理。
## 处理照片
- 处理照片主要在左侧的窗口中，一共三个页面可以切换，分别是滤镜、调节、美白磨皮，效果可以叠加，但是由于优化不好的原因，效果太多时程序会变得卡顿，建议不要对一张图片加太多效果，具体的功能的实现见另外一份文档。
## 人脸识别功能
- 在操作菜单下有人脸识别选项，点击之后会标记出图中的人脸，若想消除人脸标记效果可以点击还原选项。
## 灰度直方图功能
- 本程序实现了图像灰度直方图实时显示的功能，用户在处理图片的时候可以看到直方图的变化，这样便于调节图片到合适的效果。
## 还原照片
- 在操作菜单下有还原图片选项，恢复图片到打开初始状态。
## 保存照片
- 可以在文件菜单下找到保存图片的选项，也可以使用快捷键ctrl + s保存。
