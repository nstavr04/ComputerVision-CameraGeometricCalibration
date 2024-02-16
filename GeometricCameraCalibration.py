import cv2
import numpy as np
import glob

def resize_images(image_paths, new_size):
    """Resize images to the specified size.

    Args:
        image_paths (list of str): Paths to the images to resize.
        new_size (tuple of int): New size as (width, height).

    Returns:
        list of ndarray: List of resized images.
    """
    resized_images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        resized_img = cv2.resize(img, new_size)
        resized_images.append(resized_img)
    return resized_images

def calibrate_camera(images, square_size):
    """Calibrate camera using chessboard images.

    Args:
        images (list of ndarray): The list of images for calibration.
        square_size (float): The size of a square on the chessboard.

    Returns:
        tuple: Object points, image points, and the last processed image.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []

    for img in images:
        found, corners, processed_img = process_and_detect_corners(img, square_size, criteria)

        if found:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, (9, 6), corners, found)
            cv2.imshow('Detected Corners', img)
            cv2.waitKey(0)  
        else:
            print("Corners not found. Proceeding with manual calibration...")
            corners = manual_calibrate(img)
            if corners is not None:
                objpoints.append(objp)
                imgpoints.append(corners)
                cv2.drawChessboardCorners(img, (9, 6), np.array(corners, dtype=np.float32), True)

    cv2.destroyAllWindows()  
    return objpoints, imgpoints, processed_img

def process_and_detect_corners(img, square_size, criteria):
    """Process the image and detect chessboard corners.

    Args:
        img (ndarray): The image to process.
        square_size (float): The size of a square on the chessboard.
        criteria (tuple): The criteria for corner refinement.

    Returns:
        tuple: A tuple containing a boolean indicating if corners were found,
               the refined corners (if found), and the processed image.
    """
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # processed_img = cv2.equalizeHist(processed_img)
    
    ret, corners = cv2.findChessboardCorners(processed_img, (9, 6), None)
    
    if ret:
        corners_refined = cv2.cornerSubPix(processed_img, corners, (11, 11), (-1, -1), criteria)
        return True, corners_refined, processed_img
    else:
        return False, None, processed_img

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

    while len(corner_points) < 4:
        print("You did not select the 4 corners. Please try again.")
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

def choose_training_run():
    """Prompt user to choose a training run."""
    while True:
        choice = input("Choose training run (1, 2, or 3): ")
        if choice in ["1", "2", "3"]:
            images_path = f'Checker-Images-Train-Run{choice}/*.jpg'
            images = glob.glob(images_path)
            break
        else:
            print("Invalid choice. Please choose either 1, 2, or 3.")
    return images

def main():
    images = choose_training_run()

    square_size = 2.4

    # Read the first image to get its height and width
    img = cv2.imread(images[0])

    # Original Image Dimensions are 4000x3000
    height, width = img.shape[:2]

    # New dimensioons are 1333x1000
    new_width = width // 3
    new_height = height // 4

    ################### Camera Calibration ###################
    resized_images = resize_images(images, (new_width, new_height))

    # Call the camera calibration function - automatic calibration
    objpoints, imgpoints, gray_img = calibrate_camera(resized_images, square_size)

    # rvecs and tvecs here are only used to get the error
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)

    ##################### Online Phase #######################
    
    # We get the test image, find the corner points, and use the mtx and dist from the camera calibration and the test image points to 
    # get the rvecs and tvecs. We then use those to create the 3D axis and cube

    testing_image = cv2.imread("Checker-Images-Testing/IMG_Testing3.jpg")
    # Manually resize the training image
    resized_testing_image = cv2.resize(testing_image, (new_width, new_height))

    ret, imgpoints_test = cv2.findChessboardCorners(resized_testing_image, (9,6), None)

    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square_size

    ret, rvecs_test, tvecs_test = cv2.solvePnP(objp, imgpoints_test, mtx, dist)

    # Works without, maybe we can use them, I don't know

    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (new_width,new_height), 1, (new_width,new_height))
    # # undistort
    # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]

    # Checking the error of the calibration
    # Doesn't work for some reason when we do manual calibration
    # mean_error = 0
    # for i in range(len(objpoints)):
    #     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    #     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    #     mean_error += error
    # print( "total error: {}".format(mean_error/len(objpoints)) )

    # Draw the 3D axis and the cube on the test image
    imgpts, img_with_axes = draw_3D_axis(ret, mtx, dist, rvecs_test, tvecs_test, resized_testing_image)
    draw_3D_cube(ret, mtx, dist, rvecs_test, tvecs_test, imgpts, img_with_axes)

if __name__ == "__main__":
    main()

        