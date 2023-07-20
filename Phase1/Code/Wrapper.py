#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
from skimage.feature import peak_local_max
from scipy.ndimage.filters import maximum_filter
import cv2

# Add any python libraries here


def extract_corners(image_path, use_harris=True, block_size=2, ksize=3, k=0.04, num_corners=100):
    # Load the input image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply either Harris corner detection or Shi-Tomasi corner detection
    if use_harris:
        # Perform Harris corner detection
        corners = cv2.cornerHarris(gray, block_size, ksize, k)
    
        # Refine the corners by applying a threshold
        corners[corners < 0.001 * corners.max()] = 0
        #corners = cv2.threshold(corners, 0.001 * corners.max(), 255, cv2.THRESH_BINARY)[1]
        # Find the coordinates of the corners
        corner_coords = np.argwhere(corners > 0.01 * corners.max())
        corner_coords = [corner[::-1] for corner in corner_coords]  # Reverse x-y order

    else:
        # Perform Shi-Tomasi corner detection
        corners = cv2.goodFeaturesToTrack(gray, num_corners, 0.01, 10)
        corner_coords = np.int0(corners).reshape(-1, 2).tolist()

    # Draw the detected corners on the image
    image_with_corners = image.copy()

    if use_harris:
        # Harris corner detection produces a corner score image
        image_with_corners[corners > 0.01 * corners.max()] = [0, 0, 255]
    else:
        # Shi-Tomasi corner detection returns corner points directly
        for corner in corner_coords:
            x, y = corner
            cv2.circle(image_with_corners, (x, y), 3, (0, 0, 255), -1)

    return corners, corner_coords, image_with_corners

def ANMS(image, corner_map, N_Best=500):

    local_maxima = peak_local_max(corner_map, 15)

    N_Strong = len(local_maxima)

    r = [np.Inf for i in range(N_Strong)]

    ED = 0

    count = 0

    for i in range(N_Strong):
        for j in range(N_Strong):
            if corner_map[local_maxima[i][0]][local_maxima[i][1]] < corner_map[local_maxima[j][0]][local_maxima[j][1]]:
                ED = np.sqrt((local_maxima[i][0] - local_maxima[j][0])**2 + (local_maxima[i][1] - local_maxima[j][1])**2)
            if ED < r[i]:
                print("Inside if")
                r[i] = ED
                count += 1

    if count < N_Best:
        N_Best = count

    #Sort the r list in descending order and get the N_Best corners without using zip function
    N_Best_Corners = [local_maxima[i] for i in np.argsort(r)[::-1][:N_Best]]
  
    print(N_Best_Corners)
    
    #Show the N_Best corners on the image
    for i in range(len(N_Best_Corners)):
       cv2.circle(image, (int(N_Best_Corners[i][1]), int(N_Best_Corners[i][0])), 3, (0, 0, 255), -1)
    
    # cv2.imshow("N_Best_Corners", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("N_Best_Corners.jpg", image)  

    return N_Best_Corners  

# You need to describe each feature point by a feature vector, this is like encoding the information at each feature point by a vector. One of the easiest feature descriptor is described next.
# Take a patch of size 41×41 centered (this is very important) around the keypoint/feature point. Now apply gaussian blur (feel free to play around with the parameters, for a start you can use OpenCV’s default parameters in cv2.GaussianBlur command. Now, sub-sample the blurred output (this reduces the dimension) to 8×8
# .Then reshape to obtain a 64×1 vector.Standardize the vector to have zero mean and variance of 1.     

def feature_descriptor(image, corner_coords, patch_size=41, blur_sigma=1.5, sub_sample_size=8):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the list of feature descriptors
    feature_descriptors = []

    # Iterate over the corner coordinates
    for corner in corner_coords:
        # Extract the patch centered at the corner coordinate
        x, y = corner
        #patch = gray[int(y - patch_size // 2):int(y + patch_size // 2 + 1), int(x - patch_size // 2):int(x + patch_size // 2 + 1)]

        patch = gray[int(x - patch_size/2): int(x + patch_size/2), int(y - patch_size/2): int(y + patch_size/2)]
        # patch = cv2.GaussianBlur(patch, (3, 3), 0)

        #cv2.error: OpenCV(4.7.0) /Users/opencv-cn/GHA-OCV-1/_work/opencv-python/opencv-python/opencv/modules/imgproc/src/smooth.dispatch.cpp:617: error: (-215:Assertion failed) !_src.empty() in function 'GaussianBlur'
        #How to fix this error?

        

        if patch is None:
            raise ValueError("Image not loaded properly. Please check the image path.")

        # Apply Gaussian blur to the patch
        patch = cv2.GaussianBlur(patch, (3, 3), blur_sigma)

        # Sub-sample the blurred patch
        patch = patch[::patch_size // sub_sample_size, ::patch_size // sub_sample_size]

        # Standardize the patch
        patch = (patch - np.mean(patch)) / np.std(patch)

        # Flatten the patch into a feature vector
        feature_vector = patch.flatten()

        # Append the feature vector to the list of feature descriptors
        feature_descriptors.append(feature_vector)

    return feature_descriptors

def match_features(feature_descriptors1, feature_descriptors2, threshold=0.5):
    # Initialize the list of matching indices
    matching_indices = []

    # Iterate over the feature descriptors of the first image
    for i, feature_vector1 in enumerate(feature_descriptors1):
        # Initialize the list of distances between the feature vector and all feature vectors of the second image
        distances = []

        # Iterate over the feature descriptors of the second image
        for j, feature_vector2 in enumerate(feature_descriptors2):
            # Compute the Euclidean distance between the feature vectors
            distance = np.linalg.norm(feature_vector1 - feature_vector2)

            # Append the distance to the list of distances
            distances.append(distance)

        # Sort the list of distances in ascending order
        distances = np.argsort(distances)

        # If the ratio of the closest distance to the second closest distance is less than the threshold
        if distances[0] / distances[1] < threshold:
            # Add the matching indices to the list of matching indices
            matching_indices.append((i, distances[0]))

    return matching_indices

def estimate_homography(matching_indices, corner_coords1, corner_coords2, ransac_threshold=5):
    # Initialize the list of points in the first image
    points1 = []

    # Initialize the list of points in the second image
    points2 = []

    # Iterate over the matching indices
    for i, j in matching_indices:
        # Append the coordinates of the feature points to the lists
        points1.append(corner_coords1[i])
        points2.append(corner_coords2[j])

    # Convert the lists of points to NumPy arrays
    points1 = np.array(points1)
    points2 = np.array(points2)

    # Estimate the homography between the points using RANSAC
    homography, _ = cv2.findHomography(points1, points2, cv2.RANSAC, ransac_threshold)

    return homography

def warp_image(image, homography):
    # Warp the image using the homography
    warped_image = cv2.warpPerspective(image, homography, (image.shape[1], image.shape[0]))

    return warped_image

def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    
    detection_type = "corner-harris"
    
    filename1 = '/Users/rohin/Desktop/WPI_Fall_ 2022_Academics/RBE_549/P1/YourDirectoryID_p1/Phase1/Data/Train/Set1/1.jpg'
    img1 = cv2.imread(filename1)
    
    filename2 = '/Users/rohin/Desktop/WPI_Fall_ 2022_Academics/RBE_549/P1/YourDirectoryID_p1/Phase1/Data/Train/Set1/2.jpg'
    img2 = cv2.imread(filename2)
    
    filename3 = '/Users/rohin/Desktop/WPI_Fall_ 2022_Academics/RBE_549/P1/YourDirectoryID_p1/Phase1/Data/Train/Set1/3.jpg'
    img3 = cv2.imread(filename3)
    
    #images = [img1,img2]
    
    #cornerDetection(filename1)

    print(np.shape(img1))

    corners_1, corner_coords_1, output_image_1 = extract_corners(filename1, use_harris=True)

    best_corners_1 = ANMS(img1, corners_1, 700)

    feature_descriptors_1 = feature_descriptor(img1, best_corners_1)

    corners_2, corner_coords_2, output_image_2 = extract_corners(filename2, use_harris=True)

    best_corners_2 = ANMS(img2, corners_2, 700)

    feature_descriptors_2 = feature_descriptor(img2, best_corners_2)

    matching_indices = match_features(feature_descriptors_1, feature_descriptors_2)

    homography = estimate_homography(matching_indices, corner_coords_1, corner_coords_2)

    warped_image = warp_image(img1, homography)

    cv2.imwrite('warped_image.jpg', warped_image)

    #Display warped image
    cv2.imshow('Warped Image', warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #local_maxima = find_local_maxima(corners, neighborhood_size=5, threshold=100)
    
    #print(local_maxima)
    
    # Display the original image and the image with detected corners
    # cv2.imshow('Original Image', cv2.imread(filename1))
    # cv2.imshow('Image with Corners', output_image)  
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Print the corner coordinates
    # print("Corner Coordinates:")
    # for corner in corner_coords:
    #     print(corner)

    #print(corners)

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""

    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""

    """
	Refine: RANSAC, Estimate Homography
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
