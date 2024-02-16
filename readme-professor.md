# Computer Vision Assignment 1

## File Structure
- The image files used for training should be placed in the corresponding training folders, such as 'Checker-Images-Train-Run1'. 

- The testing images are located in the 'Checker-Images-Testing' folder. In the main function, you need to specify the name of the image in the testing folder that you want to use.

## Navigating the images. 
Please press any key when viewing the images during the run of the code to proceed to the next image.

## Manual Calibration
- When performing manual calibration, it is assumed that the points on the chessboard are clicked in the following order: 
top-left, top-right, bottom-left, bottom-right. 

- The image zooms in where you click, allowing for more precise corner selection. The interpolation of the points in manual detection is done using Homography-based interpolation.

## Choice Tasks Implemented
The following choice tasks were implemented in this assignment:

- Real-Time Performance: Webcam is used in the online phase.
- Manual Detection: The image zooms in where you click for more precise corner selection.
- Homography-based Interpolation: Interpolation of points in manual detection is done using Homography-based interpolation.

