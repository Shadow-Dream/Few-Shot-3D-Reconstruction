
from pypylon import pylon
import cv2
import numpy as np
import tqdm
SAMPLE = 0

cameras = pylon.InstantCameraArray(1)
devices = pylon.TlFactory.GetInstance().EnumerateDevices()
for i, cam in enumerate(cameras):
    cam.Attach(pylon.TlFactory.GetInstance().CreateDevice(devices[i]))
    cam.Open()
    cam.TriggerSelector.SetValue("FrameStart")
    cam.TriggerMode.SetValue("On")
    cam.TriggerSource.SetValue("Line1")
    cam.ExposureMode.SetValue("Timed")
    cam.Gain.SetValue(0)
    cam.ExposureTime.SetValue(20000)

dir = False
for camera in cameras:
    dir = not dir
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    if camera.IsGrabbing():
        grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            img = image.GetArray()
            cv2.imwrite('captures/{}_{}'.format(SAMPLE,"l" if dir else "r"), img)
    camera.StopGrabbing()
    