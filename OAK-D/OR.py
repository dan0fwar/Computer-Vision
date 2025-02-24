#!/usr/bin/env python3
import depthai as dai
import cv2
import blobconverter

labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
          "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
          "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

pipeline_instance = dai.Pipeline()

color_camera = pipeline_instance.create(dai.node.ColorCamera)
color_camera.setPreviewSize(300, 300)
color_camera.setInterleaved(False)
color_camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
color_camera.setBoardSocket(dai.CameraBoardSocket.CAM_A)

left_mono_camera = pipeline_instance.create(dai.node.MonoCamera)
left_mono_camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_1200_P)
left_mono_camera.setBoardSocket(dai.CameraBoardSocket.CAM_B)

right_mono_camera = pipeline_instance.create(dai.node.MonoCamera)
right_mono_camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_1200_P)
right_mono_camera.setBoardSocket(dai.CameraBoardSocket.CAM_C)

left_image_manip = pipeline_instance.create(dai.node.ImageManip)
left_image_manip.initialConfig.setResize(1280, 800)
left_mono_camera.out.link(left_image_manip.inputImage)

right_image_manip = pipeline_instance.create(dai.node.ImageManip)
right_image_manip.initialConfig.setResize(1280, 800)
right_mono_camera.out.link(right_image_manip.inputImage)

stereo_depth = pipeline_instance.create(dai.node.StereoDepth)
stereo_depth.initialConfig.setConfidenceThreshold(200)
stereo_depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo_depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo_depth.setRectifyEdgeFillColor(0)
stereo_depth.setLeftRightCheck(True)
stereo_depth.setSubpixel(False)

left_image_manip.out.link(stereo_depth.left)
right_image_manip.out.link(stereo_depth.right)

object_detection = pipeline_instance.create(dai.node.MobileNetSpatialDetectionNetwork)
object_detection.setBlobPath(blobconverter.from_zoo("mobilenet-ssd", shaves=6))
object_detection.setConfidenceThreshold(0.5)
object_detection.setBoundingBoxScaleFactor(0.5)
object_detection.setDepthLowerThreshold(100)
object_detection.setDepthUpperThreshold(10000)

color_camera.preview.link(object_detection.input)
stereo_depth.depth.link(object_detection.inputDepth)

output_rgb = pipeline_instance.create(dai.node.XLinkOut)
output_rgb.setStreamName("rgb")
color_camera.preview.link(output_rgb.input)

output_nn = pipeline_instance.create(dai.node.XLinkOut)
output_nn.setStreamName("nn")
object_detection.out.link(output_nn.input)

with dai.Device(pipeline_instance) as device_instance:
    print("Connected to device")
    queue_rgb = device_instance.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    queue_nn = device_instance.getOutputQueue(name="nn", maxSize=4, blocking=False)

    while True:
        rgb_input = queue_rgb.get()
        nn_input = queue_nn.get()

        frame_data = rgb_input.getCvFrame()
        detected_objects = nn_input.detections

        for detected in detected_objects:
            frame_height, frame_width = frame_data.shape[:2]
            x_min = int(detected.xmin * frame_width)
            y_min = int(detected.ymin * frame_height)
            x_max = int(detected.xmax * frame_width)
            y_max = int(detected.ymax * frame_height)

            depth_value = detected.spatialCoordinates.z
            object_distance = depth_value / 1000

            object_label = labels[detected.label] if detected.label < len(labels) else str(detected.label)
            display_text = f"{object_label} {detected.confidence:.0%} {object_distance:.2f}m"

            cv2.rectangle(frame_data, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame_data, display_text, (x_min + 10, y_min + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Object Detection", frame_data)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
