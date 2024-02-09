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
        # Not needed anymore because we use cv2.imread in image resizing
        # img = cv2.imread(fname)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            # Refines the corner positions - Not sure if it's needed or if it helps
            corners2 = cv2.cornerSubPix(gray,corners, (20,20), (-1,-1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners2, ret)
            cv2.imshow('img', img)
            # Adjust the time to see the images
            cv2.waitKey(200)

    cv2.destroyAllWindows()

    return objpoints, imgpoints, gray

# We manually calibrate the camera by clicking on the corners of the checkerboard
# The first image you see is the one you have to click on the corners
# The second image that will pop up it will show the points you clicked with red circles
# Make sure to click the corners in the following order: top-left, top-right, bottom-left, bottom-right 
def manual_calibrate(images):
    
    # Setting corner_points as global because I cannot pass parameters to the click_event function
    global corner_points
    
    for img in images:
        # Reset the corner_points for every image
        corner_points = []

        cv2.imshow("img", img)
        cv2.setMouseCallback("img", click_event, img)
        # We wait until a key is pressed
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

        cv2.imshow("img", img)
        cv2.waitKey(5000)

    cv2.destroyAllWindows()

# Used for manual calibration
def interpolate_board_points(top_left, top_right, bottom_left, bottom_right, rows=6, cols=9):

    # Generate grid coordinates
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    
    # Interpolate corner positions
    top_edge = np.linspace(top_left, top_right, cols)
    bottom_edge = np.linspace(bottom_left, bottom_right, cols)
    left_edge = np.linspace(top_left, bottom_left, rows)
    right_edge = np.linspace(top_right, bottom_right, rows)
    
    # Linear interpolation for interior points
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

# Used for manual calibration
def click_event(event, x, y, flags, params):
     if event == cv2.EVENT_LBUTTONDOWN:
        # Store the coordinates of the clicked point
        corner_points.append((x, y))
        
        # Display the point on the image for visual feedback
        cv2.circle(params, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", params)
        
        if len(corner_points) == 4:
            cv2.destroyAllWindows()

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
    cv2.waitKey(5000)
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

    images = glob.glob('Checker-Images-Training/*.jpg')
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

    # Call the manual calibration function
    # Uncomment for manual calibration
    # manual_calibrate(resized_images)

    # Call the camera calibration function - automatic calibration
    objpoints, imgpoints, gray = calibrate_camera(resized_images, square_size)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    ##################### Online Phase #######################

    training_image = cv2.imread("Checker-Images-Testing/IMG_20240209_200103.jpg")
    # Manually resize the training image
    resized_training_image = cv2.resize(training_image, (new_width, new_height))

    # Draw the 3D axis on the first image
    imgpts, img_with_axes = draw_3D_axis(ret, mtx, dist, rvecs, tvecs, resized_training_image)

    draw_3D_cube(ret, mtx, dist, rvecs, tvecs, imgpts, img_with_axes)

if __name__ == "__main__":
    main()

        