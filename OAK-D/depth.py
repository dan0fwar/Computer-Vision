#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

extend_disp = True
subpix = True
lr_chk = True
rectified_enabled = True

pipe = dai.Pipeline()

cam_left = pipe.create(dai.node.ColorCamera)
cam_center = pipe.create(dai.node.ColorCamera)
cam_right = pipe.create(dai.node.ColorCamera)
stereo_LC = pipe.create(dai.node.StereoDepth)
stereo_LR = pipe.create(dai.node.StereoDepth)
stereo_CR = pipe.create(dai.node.StereoDepth)

out_LC = pipe.create(dai.node.XLinkOut)
out_LR = pipe.create(dai.node.XLinkOut)
out_CR = pipe.create(dai.node.XLinkOut)

out_LC.setStreamName("disp_LC")
if rectified_enabled:
    out_left_LC = pipe.create(dai.node.XLinkOut)
    out_right_LC = pipe.create(dai.node.XLinkOut)
    out_left_LC.setStreamName("rect_left_LC")
    out_right_LC.setStreamName("rect_right_LC")

out_LR.setStreamName("disp_LR")
if rectified_enabled:
    out_left_LR = pipe.create(dai.node.XLinkOut)
    out_right_LR = pipe.create(dai.node.XLinkOut)
    out_left_LR.setStreamName("rect_left_LR")
    out_right_LR.setStreamName("rect_right_LR")

out_CR.setStreamName("disp_CR")
if rectified_enabled:
    out_left_CR = pipe.create(dai.node.XLinkOut)
    out_right_CR = pipe.create(dai.node.XLinkOut)
    out_left_CR.setStreamName("rect_left_CR")
    out_right_CR.setStreamName("rect_right_CR")

cam_left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
cam_left.setCamera("left")
cam_left.setIspScale(2, 3)

cam_center.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
cam_center.setBoardSocket(dai.CameraBoardSocket.CENTER)
cam_center.setIspScale(2, 3)

cam_right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
cam_right.setCamera("right")
cam_right.setIspScale(2, 3)

stereo_LC.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo_LC.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
stereo_LC.setLeftRightCheck(lr_chk)
stereo_LC.setExtendedDisparity(extend_disp)
stereo_LC.setSubpixel(subpix)

stereo_LR.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo_LR.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
stereo_LR.setLeftRightCheck(lr_chk)
stereo_LR.setExtendedDisparity(extend_disp)
stereo_LR.setSubpixel(subpix)

stereo_CR.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo_CR.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
stereo_CR.setLeftRightCheck(lr_chk)
stereo_CR.setExtendedDisparity(extend_disp)
stereo_CR.setSubpixel(subpix)

cam_left.isp.link(stereo_LC.left)
cam_center.isp.link(stereo_LC.right)
stereo_LC.disparity.link(out_LC.input)
if rectified_enabled:
    stereo_LC.rectifiedLeft.link(out_left_LC.input)
    stereo_LC.rectifiedRight.link(out_right_LC.input)

cam_left.isp.link(stereo_LR.left)
cam_right.isp.link(stereo_LR.right)
stereo_LR.disparity.link(out_LR.input)
if rectified_enabled:
    stereo_LR.rectifiedLeft.link(out_left_LR.input)
    stereo_LR.rectifiedRight.link(out_right_LR.input)

cam_center.isp.link(stereo_CR.left)
cam_right.isp.link(stereo_CR.right)
stereo_CR.disparity.link(out_CR.input)
if rectified_enabled:
    stereo_CR.rectifiedLeft.link(out_left_CR.input)
    stereo_CR.rectifiedRight.link(out_right_CR.input)

max_disparity = stereo_LC.initialConfig.getMaxDisparity()

with dai.Device(pipe) as dev:
    while not dev.isClosed():
        queue_names = dev.getQueueEvents()
        for queue in queue_names:
            msg = dev.getOutputQueue(queue).get()
            if type(msg) == dai.ImgFrame:
                frm = msg.getCvFrame()
                if 'disp' in queue:
                    disp_map = (frm * (255.0 / max_disparity)).astype(np.uint8)
                    disp_map = cv2.applyColorMap(disp_map, cv2.COLORMAP_JET)
                    cv2.imshow(queue, disp_map)
                else:
                    cv2.imshow(queue, frm)
        if cv2.waitKey(1) == ord('q'):
            break
