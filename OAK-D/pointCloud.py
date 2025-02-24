#!/usr/bin/env python3
import numpy as np
import depthai as dai
import depthai_viewer as viewer
import subprocess
import time
import sys
import cv2

subprocess.Popen([sys.executable, "-m", "depthai_viewer"])
time.sleep(1)
viewer.init("Depthai Viewer")
viewer.connect()

def create_pipeline():
    pipeline_instance = dai.Pipeline()

    rgb_camera = pipeline_instance.create(dai.node.ColorCamera)
    rgb_camera.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    rgb_camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    rgb_camera.setIspScale(1, 2)
    rgb_camera.setInterleaved(False)
    rgb_camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    right_camera = pipeline_instance.create(dai.node.MonoCamera)
    right_camera.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right_camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)

    stereo_depth = pipeline_instance.create(dai.node.StereoDepth)
    stereo_depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo_depth.setLeftRightCheck(True)
    stereo_depth.setExtendedDisparity(False)
    stereo_depth.setSubpixel(True)
    stereo_depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo_depth.setOutputSize(640, 400)

    rgb_camera.isp.link(stereo_depth.left)
    right_camera.out.link(stereo_depth.right)

    point_cloud = pipeline_instance.create(dai.node.PointCloud)
    stereo_depth.depth.link(point_cloud.inputDepth)

    rgb_output = pipeline_instance.create(dai.node.XLinkOut)
    rgb_output.setStreamName("rgb")
    rgb_camera.isp.link(rgb_output.input)

    disparity_output = pipeline_instance.create(dai.node.XLinkOut)
    disparity_output.setStreamName("disparity")
    stereo_depth.disparity.link(disparity_output.input)

    point_cloud_output = pipeline_instance.create(dai.node.XLinkOut)
    point_cloud_output.setStreamName("pcl")
    point_cloud.outputPointCloud.link(point_cloud_output.input)

    return pipeline_instance

def main():
    print("Starting pipeline...")
    with dai.Device() as device_instance:
        pipeline_instance = create_pipeline()
        device_instance.startPipeline(pipeline_instance)

        queue_rgb = device_instance.getOutputQueue("rgb", maxSize=4, blocking=False)
        queue_disparity = device_instance.getOutputQueue("disparity", maxSize=4, blocking=False)
        queue_point_cloud = device_instance.getOutputQueue("pcl", maxSize=4, blocking=False)

        max_disparity = 95

        while True:
            frame_rgb = queue_rgb.get().getCvFrame()
            frame_disparity = queue_disparity.get().getFrame()
            point_cloud_data = queue_point_cloud.get()

            colored_disparity = cv2.applyColorMap(
                (frame_disparity * (255 / max_disparity)).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            point_array = np.array(point_cloud_data.points).reshape(-1, 3)
            if point_array.size > 0:
                height, width = frame_rgb.shape[:2]
                y_values = np.clip((point_array[:, 1] + 1) * height/2, 0, height-1).astype(int)
                x_values = np.clip((point_array[:, 0] + 1) * width/2, 0, width-1).astype(int)
                color_values = frame_rgb[y_values, x_values] / 255.0

                viewer.log_points("PointCloud", point_array, colors=color_values)

            viewer.log_image("RGB", frame_rgb)
            viewer.log_image("Disparity", colored_disparity)

            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
