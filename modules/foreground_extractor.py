import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class ForegroundExtractor:
    def __init__(self,kernel_size = 3):
        self.extractor = cv.bgsegm.BackgroundSubtractorGSOC()
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

    def get_foreground_mask(self,image):
        mask = self.extractor.apply(image)
        mask = cv.morphologyEx(mask,cv.MORPH_OPEN,self.kernel)
        return mask

    def get_foreground_image(self,image):
        mask = self.get_foreground_mask(image)
        return image

extractor = ForegroundExtractor()
image = cv.imread("D:/A_Code/Classes/IAVI2023/chessboard/0_l.jpg")
extractor.get_foreground_image(image)