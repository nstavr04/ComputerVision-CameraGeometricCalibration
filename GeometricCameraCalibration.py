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
    return [cv2.resize(cv2.imread(image_path), new_size) for image_path in image_paths]

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
            corners = manual_calibrate(img, square_size)
            if corners is not None:
                objpoints.append(objp)
                imgpoints.append(corners)
                cv2.drawChessboardCorners(img, (9, 6), np.array(corners, dtype=np.float32), True)
                cv2.imshow('Detected Corners', img)
                cv2.waitKey(0)

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
    
    # We make the image grayscale, apply GaussianBlur and then CLAHE to help with the automatic corner detection
    # Overall, it lowers the total error of the calibration
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(6, 9))
    processed_img = clahe.apply(processed_img)
    
    ret, corners = cv2.findChessboardCorners(processed_img, (9, 6), None)
    
    if ret:
        print("Found the corners in the image. Proceeding with automatic calibration...")
        corners_refined = cv2.cornerSubPix(processed_img, corners, (11, 11), (-1, -1), criteria)
        return True, corners_refined, processed_img
    else:
        print("Could not find the corners in the image. Proceeding with manual calibration...")
        return False, None, processed_img
    
# We manually calibrate the camera by clicking on the corners of the checkerboard
# The first image you see is the one you have to click on the corners
# The second image that will pop up it will show the points you clicked with red circles
# Make sure to click the corners in the following order: top-left, top-right, bottom-left, bottom-right 
def manual_calibrate(img, square_size):
    
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

    all_points = interpolate_board_points_homography(top_left, top_right, bottom_left, bottom_right, square_size)
    
    # Draw the points on the image
    for point in all_points:
        cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    
    # Ensure correct shape (N, 1, 2) - Was needed otherwise in error calculation in main we would get an error
    all_points = all_points.reshape(-1, 1, 2)

    return all_points

# Used for manual calibration
def interpolate_board_points_homography(top_left, top_right, bottom_left, bottom_right, square_size=2.4, rows=6, cols=9):
    
    # Points - image plane
    dst_pts = np.array([top_left, top_right, bottom_left, bottom_right], dtype="float32")

    # Points - checkerboard plane
    src_pts = np.array([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]], dtype="float32") * square_size

    H, _ = cv2.findHomography(src_pts, dst_pts)

    # Grid points on the checkerboard
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    checkerboard_pts = np.hstack((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1))) * square_size
    checkerboard_pts_homogeneous = np.insert(checkerboard_pts, 2, 1, axis=1).T

    # Show the grid points in the image
    image_pts_homogeneous = np.dot(H, checkerboard_pts_homogeneous)
    image_pts = image_pts_homogeneous[:2, :] / image_pts_homogeneous[2, :]
    
    image_pts = image_pts.T.reshape(-1, 2).astype(np.float32)

    return image_pts

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

def calculate_total_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        return mean_error/len(objpoints)

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

def start_live_camera(mtx, dist, square_size=2.4):
    # Capture live video feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Attempt to find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        
        if ret:
            # Refine the corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
            # Estimate the pose of the chessboard
            objp = np.zeros((6*9,3), np.float32)
            objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) * square_size
            _, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist, None, None)

            # Draw 3D axis and cube
            frame = draw_3D_axis_live_camera(frame, corners2[0].ravel(), mtx, dist, rvecs, tvecs)

        # Display the resulting frame
        cv2.imshow('Live', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def draw_3D_axis_live_camera(img, corner, mtx, dist, rvecs, tvecs):
    # Define the 3D points for axis (3 units long).
    axis = np.float32([[0,0,0], [3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

    # Project 3D points to 2D image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

    # Draw axis lines
    corner = tuple(corner.ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[3].ravel().astype(int)), (0,0,255), 5)

    # Define 8 corners of the cube in 3D space (assuming cube size of 3 units).
    cube_size = 3
    cube = np.float32([[0,0,0], [0,cube_size,0], [cube_size,cube_size,0], [cube_size,0,0],
                    [0,0,-cube_size], [0,cube_size,-cube_size], [cube_size,cube_size,-cube_size], [cube_size,0,-cube_size]])

    # Project cube points to the 2D image plane
    imgpts_cube, _ = cv2.projectPoints(cube, rvecs, tvecs, mtx, dist)

    # Draw the cube by connecting the projected points
    imgpts_cube = imgpts_cube.reshape(-1, 2).astype(int)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting top and bottom faces
    ]
    for start, end in edges:
        img = cv2.line(img, tuple(imgpts_cube[start]), tuple(imgpts_cube[end]), (0,255,255), 3)

    return img

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

    # Resize the images
    resized_images = resize_images(images, (new_width, new_height))

    # Call the camera calibration function - automatic calibration
    objpoints, imgpoints, gray_img = calibrate_camera(resized_images, square_size)

    # rvecs and tvecs here are only used to get the error
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)

    ##################### Online Phase #######################
    
    # We get the test image, find the corner points, and use the mtx and dist from the camera calibration and the test image points to 
    # get the rvecs and tvecs. We then use those to create the 3D axis and cube

    # !!!!!!!! CHANGE ACCORDINGLY TO THE TEST IMAGE YOU WANT TO USE !!!!!!!!
    testing_image = cv2.imread("Checker-Images-Testing/IMG_Testing3.jpg")

    resized_testing_image = cv2.resize(testing_image, (new_width, new_height))

    ret, imgpoints_test = cv2.findChessboardCorners(resized_testing_image, (9,6), None)

    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square_size

    ret, rvecs_test, tvecs_test = cv2.solvePnP(objp, imgpoints_test, mtx, dist)

    # Checking the error of the calibration
    total_error = calculate_total_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    print("total error: {}".format(total_error))

    # Draw the 3D axis and the cube on the test image
    imgpts, img_with_axes = draw_3D_axis(ret, mtx, dist, rvecs_test, tvecs_test, resized_testing_image)
    draw_3D_cube(ret, mtx, dist, rvecs_test, tvecs_test, imgpts, img_with_axes)
    start_live_camera(mtx, dist)

if __name__ == "__main__":
    main()
