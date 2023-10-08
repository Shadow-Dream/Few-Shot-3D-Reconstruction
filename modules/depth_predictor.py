import cv2 as cv
import numpy as np
import math

# 通过双目预测两张图片对应的深度图
# baseline 基线长度，两个相机之间的水平距离（相机垂直距离为0，因为都固定在架子上）
# focal 相机焦距
# intrinsic_matrix 相机内参矩阵
# distortion 相机畸变参数
# image_width 图像裁剪宽度（将输入裁剪成这个大小）
# image_height 图像裁剪高度
# pyramid_num 图像金字塔高度（图像金字塔就是将图像缩放成不同大小，多次执行算法，可以提高准确度）
# disparities_num 计算图像视差时候的最大搜索范围，所谓视差就是同一个真实世界位置对应的像素点在左图和右图中的x坐标之差
# disparities_block_size 计算图像视差时候的搜索块大小，块越大越平滑，准确度越低，错误率也越低
# gaussian_kernel_size 图像去噪时候使用的高斯核大小
class DepthPredictor:
    def __init__(self,baseline,focal,intrinsic_matrix,distortion,
    image_width = 1920,image_height = 1080,pyramid_num = 6,
    disparities_num = 16,disparities_block_size = 15,
    gaussian_kernel_size = 15):
        #初始化成员变量
        self.image_width = image_width
        self.image_height = image_height
        self.baseline = baseline
        self.focal = focal
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion = distortion
        self.pyramid_num = pyramid_num
        self.disparities_num = disparities_num
        self.disparities_block_size = disparities_block_size
        self.gaussian_kernel_size = gaussian_kernel_size

    def clip(self, image):
        #将图像修剪为指定大小
        min_height = (height - self.image_height)//2
        max_height = (height + self.image_height)//2
        min_width = (width - self.image_width)//2
        max_width = (width + self.image_width)//2
        height,width = image.shape[0],image.shape[1]
        image = image[min_height:max_height,min_width:max_width]
        return image

    def preprocess(self, image):
        #预处理图像，使用畸变参数来纠正图像畸变，然后高斯模糊去噪
        image = cv.undistort(image, self.intrinsic_matrix, self.distortion)
        image = cv.GaussianBlur(image,(self.gaussian_kernel_size,self.gaussian_kernel_size),0,0)
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        return image

    def generate_pyramid(self,image):
        #生成图像金字塔，就是生成一个数组，里面由大到小地存着不同大小的图像，每一张图比上一张小一倍
        pyramid = []
        for i in range(self.pyramid_num):
            scale = 2 ** i
            scaled_image = cv.resize(image,(self.image_width//scale,self.image_height//scale))
            pyramid.append(scaled_image)
        return pyramid
    
    def _estimate_disparity(self,image_left,image_right):
        #计算image_left和image_right之间的视差图，由于left对right有视差，right对left也有视差，所以返回两个
        stereo = cv.StereoBM_create(numDisparities=self.disparities_num, blockSize=self.disparities_block_size)
        disparity_left = stereo.compute(image_left, image_right)
        disparity_right = stereo.compute(image_right, image_left)
        return disparity_left,disparity_right

    def sum_pyramid(self,pyramid):
        #将金字塔中的所有图片加起来，可能会用到这个函数，先写着
        sum = np.zeros_like(pyramid[0])
        for image in pyramid:
            image = cv.resize(image,(self.image_width,self.image_height))
            sum += image
        return sum
    
    #计算image_left和image_right之间的视差图金字塔，也就是对image_left和image_right图像金字塔中的每一层都计算视差
    def estimate_disparity_pyramid(self,image_left,image_right):
        pyramid_left = self.generate_pyramid(image_left)
        pyramid_right = self.generate_pyramid(image_right)
        pyramid_disparity_left = []
        pyramid_disparity_right = []
        for image_left,image_right in zip(pyramid_left,pyramid_right):
            disparity_left,disparity_right = self._estimate_disparity(image_left,image_right)
            pyramid_disparity_left.append(disparity_left)
            pyramid_disparity_right.append(disparity_right)
        return pyramid_disparity_left,pyramid_disparity_right
    
    #计算image_left和image_right计算深度图
    #1. 视差和深度之间有转换公式，需要用到内参矩阵，焦距和基线长度
    #2. 不同大小的图片，内参矩阵等参数会发生变化，视差的大小可能也会变化，需要加以转换，最后才能sum到一起获得更准确的结果（图像金字塔每一层都要计算）
    #3. 这里输入的是原始的图像，不是剪裁过的，因此需要先剪裁、预处理再用（剪裁的原因是畸变参数可能算的不太好，导致图像边缘畸变仍然比较严重）
    # image_left = self.clip(self.preprocess(image_left))
    # image_right = self.clip(self.preprocess(image_right))
    #4. 这个函数要返回左图和右图分别对应的深度
    def estimate_depth(self,image_left,image_right):
        #TODO
        return