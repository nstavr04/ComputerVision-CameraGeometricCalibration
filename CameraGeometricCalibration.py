import cv2
import numpy as np
import glob

# Im resizing the images because when we do cv2.imshow in the calibration function, the images are super zoomed in
def resize_images(images, new_width, new_height):
        resized_images = []
        for image in images:
            img2 = cv2.imread(image)
            resized_img = cv2.resize(img2, (new_width, new_height))
            resized_images.append(resized_img)
        return resized_images

def calibrate_camera(images, square_size):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Some of the below code was taken from here:
    # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

    # It's to prepare object points (our board is 7x10 so we want 6x9 points I think)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for img in images:

        # We do the following on the image to improve the accuracy of the corner detection:
        # Turning image to grayscale
        # Enhancing edges -> Cany Edge Detection
        # For getting rid of light reflections -> Histogram Equalization
        
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.equalizeHist(processed_img)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(processed_img, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            print("Found the corners in the image. Proceeding with automatic calibration...")

            objpoints.append(objp)

            # Refines the corner positions - findChessboardCorners uses this function,
            # but we can use it with different parameters to get better results
            corners2 = cv2.cornerSubPix(processed_img, corners, (20,20), (-1,-1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners2, ret)
            cv2.imshow('img', img)
            # Wait for key press
            cv2.waitKey(0)
        
        # If we didn't find the corners, we will do manual calibration
        else:
            print("Could not find the corners in the image. Proceeding with manual calibration...")
            
            corners = manual_calibrate(img)
            objpoints.append(objp)
            imgpoints.append(corners)

            # We set to true to be able to use the corner drawing function
            ret = True
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    return objpoints, imgpoints, processed_img

# We manually calibrate the camera by clicking on the corners of the checkerboard
# The first image you see is the one you have to click on the corners
# The second image that will pop up it will show the points you clicked with red circles
# Make sure to click the corners in the following order: top-left, top-right, bottom-left, bottom-right 
def manual_calibrate(img):
    
    # Setting corner_points as global because I cannot pass parameters to the click_event function
    global corner_points, original_image

    # Reset the corner_points for every image
    corner_points = []

    cv2.imshow("img", img)
    original_image = img
    cv2.setMouseCallback("img", click_event, img)
    cv2.waitKey(0)

    # We assume that the corner_points are in the following order:
    # top-left, top-right, bottom-left, bottom-right
    top_left = corner_points[0]
    top_right = corner_points[1]
    bottom_left = corner_points[2]
    bottom_right = corner_points[3]

    all_points = interpolate_board_points(top_left, top_right, bottom_left, bottom_right)
    
    # Draw the points on the image
    for point in all_points:
        cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    
    return all_points

# Used for manual calibration
def interpolate_board_points(top_left, top_right, bottom_left, bottom_right, rows=6, cols=9):

    # Generate grid coordinates
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    
    # We want to interpolate corner positions
    top_edge = np.linspace(top_left, top_right, cols)
    bottom_edge = np.linspace(bottom_left, bottom_right, cols)
    left_edge = np.linspace(top_left, bottom_left, rows)
    right_edge = np.linspace(top_right, bottom_right, rows)
    
    interior_points = []
    for y in range(rows):
        for x in range(cols):
            # Interpolate points along the top and bottom edges
            top = top_edge[x]
            bottom = bottom_edge[x]
            
            # Finally, interpolate between the top and bottom to find the point
            point = top * (1 - grid_y[y, x]) + bottom * grid_y[y, x]
            interior_points.append(point)
            
    return np.array(interior_points, dtype=np.float32)

def click_event(event, x, y, flags, params):

    global zoom_scale, zoom_window_size
    zoom_scale = 4
    zoom_window_size = 200
    
    if event == cv2.EVENT_LBUTTONDOWN and not original_image is None:
        # Calculate bounds for the zoomed region
        x_min = max(x - zoom_window_size, 0)
        y_min = max(y - zoom_window_size, 0)
        x_max = min(x + zoom_window_size, original_image.shape[1])
        y_max = min(y + zoom_window_size, original_image.shape[0])

        # Extract and zoom in on the region
        zoom_region = original_image[y_min:y_max, x_min:x_max].copy()
        zoom_region = cv2.resize(zoom_region, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_LINEAR)
        
        # Display the zoomed window
        cv2.imshow("Zoomed", zoom_region)
        cv2.setMouseCallback("Zoomed", click_event_zoomed, (x_min, y_min, x_max, y_max))

def click_event_zoomed(event, x, y, flags, params):
    global corner_points, original_image

    if event == cv2.EVENT_LBUTTONDOWN and not original_image is None:
        # Translate click position back to original image coordinates
        x_min, y_min, x_max, y_max = params
        precise_x = int(x / zoom_scale + x_min)
        precise_y = int(y / zoom_scale + y_min)

        # Append precise corner point
        corner_points.append((precise_x, precise_y))
        
        # Visual feedback on the original image
        cv2.circle(original_image, (precise_x, precise_y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", original_image)
        
        if len(corner_points) == 4:
            original_image = None
            cv2.destroyAllWindows()
        else:
            cv2.destroyWindow("Zoomed")

def draw_3D_axis(ret, mtx, dist, rvecs, tvecs, training_image):
    # Axis points in 3D space. We'll draw the axis lines from the origin to these points.
    # Increasing the numbers makes the lines longer
    axis = np.float32([[0, 0, 0], [9, 0, 0], [0, 9, 0], [0, 0, -9]]).reshape(-1, 3)

    img_with_axes = training_image.copy()
    rvecs, tvecs = rvecs[0], tvecs[0]

    # Project the 3D points to the 2D image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    imgpts = imgpts.astype(int)

    # Define the origin (chessboard corner in this case)
    origin = tuple(imgpts[0].ravel())
    img_with_axes = cv2.line(img_with_axes, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 5)  # X-Axis in red
    img_with_axes = cv2.line(img_with_axes, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 5)  # Y-Axis in green
    img_with_axes = cv2.line(img_with_axes, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 5)  # Z-Axis in blue

    cv2.imshow('Image_with_axes', img_with_axes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return imgpts, img_with_axes

def draw_3D_cube(ret, mtx, dist, rvecs, tvecs, imgpts, img_with_axes):

    # Define a cube with side length 2, centered at the origin
    side_length = 5
    cube = np.float32([
        [0, 0, 0],  # Bottom-back-left at origin
        [side_length, 0, 0],  # Bottom-back-right
        [side_length, side_length, 0],  # Bottom-front-right
        [0, side_length, 0],  # Bottom-front-left
        [0, 0, -side_length],  # Top-back-left
        [side_length, 0, -side_length],  # Top-back-right
        [side_length, side_length, -side_length],  # Top-front-right
        [0, side_length, -side_length]  # Top-front-left
    ])

    # Project the cubes corners on the 2D image plane
    rvecs, tvecs = rvecs[0], tvecs[0]
    imgpts_cube, _ = cv2.projectPoints(cube, rvecs, tvecs, mtx, dist)
    imgpts_cube = imgpts_cube.reshape(-1, 2).astype(int)

    # Edges of the cube
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges connecting top and bottom faces
    ]

    # Draw the edges
    for start, end in edges:
        img_with_axes = cv2.line(img_with_axes, tuple(imgpts_cube[start]), tuple(imgpts_cube[end]), (0, 165, 255), 3)

    img_with_cube = img_with_axes
    cv2.imshow('Image with Cube', img_with_cube)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():

    while True:
        run_choice = input("Choose training run (1, 2, or 3): ")
        
        if run_choice == "1":
            print("Training with all 25 images - 5 of them were manually calibrated")
            images = glob.glob('Checker-Images-Train-Run1/*.jpg')
            break
        elif run_choice == "2":
            print("Training with 10 images - all of them were automatically calibrated")
            images = glob.glob('Checker-Images-Train-Run2/*.jpg')
            break
        elif run_choice == "3":
            print("Training with 5 images - all of them were automatically calibrated")
            images = glob.glob('Checker-Images-Train-Run3/*.jpg')
            break
        else:
            print("Invalid choice. Please choose either 1, 2, or 3.")

    square_size = 2.4

    # Read the first image to get its height and width
    img = cv2.imread(images[0])

    # Original Image Dimensions are 4000x3000
    height, width = img.shape[:2]

    # New dimensioons are 1333x1000
    new_width = width // 3
    new_height = height // 4

    ################### Camera Calibration ###################

    # Resize the images
    resized_images = resize_images(images, new_width, new_height)

    # Call the camera calibration function - automatic calibration
    objpoints, imgpoints, gray_img = calibrate_camera(resized_images, square_size)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)

    ##################### Online Phase #######################

    testing_image = cv2.imread("Checker-Images-Testing/IMG_Testing2.jpg")
    # Manually resize the training image
    resized_testing_image = cv2.resize(testing_image, (new_width, new_height))

    # Draw the 3D axis on the first image
    imgpts, img_with_axes = draw_3D_axis(ret, mtx, dist, rvecs, tvecs, resized_testing_image)

    draw_3D_cube(ret, mtx, dist, rvecs, tvecs, imgpts, img_with_axes)

if __name__ == "__main__":
    main()

        