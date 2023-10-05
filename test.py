import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import torch

MIN_MATCH_COUNT = 10

device = "cuda"

def compute_match_keypoints(image_ref,image_src):
    image_ref = cv.cvtColor(image_ref,cv.COLOR_BGR2GRAY)
    image_src = cv.cvtColor(image_src,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT.create()

    keypoints_ref = sift.detect(image_ref)
    keypoints_ref,features_ref = sift.compute(image_ref,keypoints_ref)

    keypoints_src = sift.detect(image_src)
    keypoints_src,feature_src = sift.compute(image_src,keypoints_src)

    index_params = dict(algorithm = 1, trees = 5)
    search_params = dict(checks=50)
    matcher = cv.FlannBasedMatcher(index_params,search_params)

    matches = matcher.knnMatch(features_ref,feature_src,k=2)
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)
    matches = good_matches
    match_points_ref = np.array([keypoints_ref[match.queryIdx].pt for match in matches])
    match_points_src = np.array([keypoints_src[match.trainIdx].pt for match in matches])
    return match_points_ref,match_points_src

def keypoint_correspondence_fliter(match_points_ref,match_points_src):
    match_points_ref = match_points_ref.reshape(-1,1,2)
    match_points_src = match_points_src.reshape(-1,1,2)

    homography_matrix,mask = cv.findHomography(match_points_ref,match_points_src,cv.RANSAC)

    mask = mask.squeeze()
    match_points_ref = match_points_ref.squeeze()
    match_points_src = match_points_src.squeeze()
    match_points_ref = np.compress(mask==1,match_points_ref,0)
    match_points_src = np.compress(mask==1,match_points_src,0)

    return homography_matrix,match_points_ref,match_points_src

def surface_area_fliter(match_points_ref,match_points_src):
    return match_points_ref,match_points_src

def dense_verification_fliter(match_points_ref,match_points_src):
    return match_points_ref,match_points_src