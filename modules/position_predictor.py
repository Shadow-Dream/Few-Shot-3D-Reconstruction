import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

# 用来估计两张图片之间的投影矩阵（projection matrix）
# 按理来说棋盘格那份代码里面就有相关的实现，他那个函数最后能够输出相机相对于棋盘格的位置和旋转角，其实就是计算投影矩阵
class PositionPredictor:
    def __init__(self,intrinsic_matrix,min_match_ratio = 0.7):
        self.intrinsic_matric = intrinsic_matrix
        self.min_match_ratio = min_match_ratio
        
    # 计算两张图像之间匹配的关键点和它们的位置（SIFT角点检测，SURF垃圾算法不开源，这里再骂一遍）
    def compute_match_keypoints(self, image_ref,image_src):
        image_ref = cv.cvtColor(image_ref,cv.COLOR_BGR2GRAY)
        image_src = cv.cvtColor(image_src,cv.COLOR_BGR2GRAY)
        sift = cv.SIFT.create()

        #检测两张图像的关键点与关键点的特征向量
        keypoints_ref = sift.detect(image_ref)
        keypoints_ref,features_ref = sift.compute(image_ref,keypoints_ref)

        keypoints_src = sift.detect(image_src)
        keypoints_src,feature_src = sift.compute(image_src,keypoints_src)

        #使用knn进行特征向量匹配，找出两张图像中对应的关键点（理解为现实世界的同一点）
        index_params = dict(algorithm = 1, trees = 5)
        search_params = dict(checks=50)
        matcher = cv.FlannBasedMatcher(index_params,search_params)
        matches = matcher.knnMatch(features_ref,feature_src,k=2)

        #筛选掉匹配得不好的点
        good_matches = []
        for m,n in matches:
            if m.distance < self.min_match_ratio*n.distance:
                good_matches.append(m)
        matches = good_matches

        #返回能匹配到的点坐标，match_points_ref和match_points_src分别为关键点在两张图中的二维坐标
        match_points_ref = np.array([keypoints_ref[match.queryIdx].pt for match in matches])
        match_points_src = np.array([keypoints_src[match.trainIdx].pt for match in matches])
        return match_points_ref,match_points_src
    
    # 继续筛选关键点，将不符合图像映射关系的关键点全部去除掉
    # 这里的原理是利用所有关键点计算出两张图片近似的单应性矩阵H（假设A图片上，M物体对应的点坐标是p，那么B图片上M物体对应的点坐标就是Hp）
    # 近似的单应性矩阵可以通过最小二乘等方法求出
    # 将所有A图中特征点左乘单应性矩阵投影到B图上，如果和B图对应的关键点距离太远，可以认为是错误匹配，则去除掉
    # 这个函数返回单应性矩阵和筛选后的特征点
    def keypoint_correspondence_fliter(self,match_points_ref,match_points_src):
        match_points_ref = match_points_ref.reshape(-1,1,2)
        match_points_src = match_points_src.reshape(-1,1,2)

        homography_matrix,mask = cv.findHomography(match_points_ref,match_points_src,cv.RANSAC)

        mask = mask.squeeze()
        match_points_ref = match_points_ref.squeeze()
        match_points_src = match_points_src.squeeze()
        match_points_ref = np.compress(mask==1,match_points_ref,0)
        match_points_src = np.compress(mask==1,match_points_src,0)

        return homography_matrix,match_points_ref,match_points_src
    
    # 另外一个特征点过滤器，懒得实现了，到时候再说
    def surface_area_fliter(self,match_points_ref,match_points_src):
        return match_points_ref,match_points_src

    # 另外一个特征点过滤器，懒得实现了，到时候再说
    def dense_verification_fliter(self,match_points_ref,match_points_src):
        return match_points_ref,match_points_src
    
    # 计算image_src到image_ref的投影矩阵projection matrix
    # 投影矩阵可以通过单应性矩阵和内参矩阵求出
    def estimate_projection_matrix(self,image_ref,image_src):
        #TODO
        return