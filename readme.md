# Chessboard Calibration Project

This project is divided into two main phases: the offline phase and the online phase.

## Offline Phase

1. **Print the Chessboard Image:** Print the provided chessboard image. Measure the size of each cell and figure out where you have to use this in your code.

2. **Implement Corner Detection Interface:** While OpenCV provides a function to find chessboard corners, it sometimes fails. Implement an interface that asks the user to manually provide the four corner points and linearly interpolates all chessboard points from those coordinates. The output should be similar to the OpenCV function.

3. **Capture Training Images:** With a camera or webcam, take 25 training images of the chessboard in different positions and orientations. At least 5 images should be where OpenCV could not detect the corners. These images require manual annotation of the corner points. Take a final test image that has the chessboard tilted significantly, and close to the image border.

4. **Geometric Camera Calibration:** Implement the geometric-camera calibration using OpenCV functions. Calibrate your camera using the training images. The camera center should not be fixed but estimated and you can have different focal lengths in horizontal and vertical directions. Perform three runs of calibration:
    - Run 1: Use all training images.
    - Run 2: Use only ten images with automatically found corner points.
    - Run 3: Use only five out of the ten images in Run 2.

## Online Phase

For each run, take the test image and draw the world 3D axes (XYZ) with the origin at the center of the world coordinates, using the estimated camera parameters. Also draw a cube which is located at the origin of the world coordinates. Bonus points are awarded for real-time execution using your webcam.

## Code

You can develop a script for the online and offline phase independently or combined. Please write comments at the beginning of every function to explain its purpose.

## Report

Your report should be around 1-2 pages and contain:
- For each of the three runs, the intrinsic camera matrix and an example screenshot with axes and cube.
- Provide the explicit form of camera intrinsics matrix K.
- A brief explanation of how the estimation of each element in the intrinsics matrix changes for the three runs.
- A brief mention of the choice tasks that you've done, and how you implemented them.

## Grading

The maximum score for this assignment is 100 (grade 10). The assignment counts for 10% of the course grade. You can get 70 regular points and a maximum of 30 points for chosen tasks:

### Regular Tasks
- **Offline Calibration Stage:** 20 points
- **Offline Manual Corner Annotation Interface:** When no corners are found automatically: 15 points
- **Online Stage with Cube Drawing:** 20 points
- **Reporting:** 10 points
- **Screen-Snapshots:** Correct and accurate: 5 points

### Choice Tasks
- **Real-Time Performance:** With webcam in online phase: 10 points
- **Iterative Detection and Rejection:** Of low quality input images in offline phase. Check for a function output that could provide an indication of the quality of the calibration: 10 points
- **Improving the Localization of the Corner Points:** In your manual interface. Make the estimation of (a) the four corner points or (b) the interpolated points more accurate: 10 points
- **Animation/Shading/Etc.:** That demonstrates that you can project 3D to 2D. There needs to be explicit depth reasoning so lines/vertices further away should not overlap nearer ones: 10 points
- **Implement a Function:** That can provide a confidence for how well each variable has been estimated, perhaps by considering subsets of the input: 10 points
- **Enhance the Input:** To reduce the number of input images that are not correctly processed by `findChessboardCorners`, for example by enhancing edges or getting rid of light reflections: 10 points
- **Produce a 3D Plot:** With the locations of the camera relative to the chessboard when each of the training images was taken: 10 points