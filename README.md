# OAK-D Object Recognition, Depth Processing, and Pointcloud Generation

This repository contains Python scripts for object recognition, distance measurement, point cloud generation, and depth mapping using the OAK-D LR camera.


##  Features

- **`OR.py`**: Detects objects and calculates their distance using depth information from the OAK-D camera.
- **`pointcloud.py`**: Generates a 3D point cloud representation from depth data.
- **`depth.py`**: Visualizes the depth map captured by the OAK-D camera.

##  Requirements

Ensure you have the following installed:

- **Python 3.x**  
- **depthai** – for interfacing with the OAK-D camera  
- **OpenCV (`cv2`)** – for image processing  
- **blobconverter** – for converting AI models  
- **numpy** – for numerical computations  
- **depthai_viewer** – for visualizing depth data  
- **subprocess, time, sys** – standard Python libraries  

Install dependencies using:

```bash
pip install depthai opencv-python blobconverter numpy depthai_viewer

