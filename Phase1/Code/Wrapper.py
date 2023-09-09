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
                r[i] = ED
                count += 1

    if count < N_Best:
        N_Best = count

    #Sort the r list in descending order and get the N_Best corners without using zip function
    N_Best_Corners = [local_maxima[i] for i in np.argsort(r)[::-1][:N_Best]]
  
    #print(N_Best_Corners)
    
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

def feature_descriptor(image, corner_coords, patch_size=41, blur_sigma=1.5):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width1, height1 = gray.shape

    # Initialize the list of feature descriptors
    feature_descriptors = []

    # Iterate over the corner coordinates
    for corner in corner_coords:
        # Extract the patch centered at the corner coordinate
        x, y = corner
        #patch = gray[int(y - patch_size // 2):int(y + patch_size // 2 + 1), int(x - patch_size // 2):int(x + patch_size // 2 + 1)]

        if ((int(x - patch_size/2) > 0) and (int(x + patch_size/2) < height1)) and ((int(y - patch_size/2) > 0) and (int(y + patch_size/2) < width1)):
            patch = gray[int(y - patch_size/2): int(y + patch_size/2), int(x - patch_size/2): int(x + patch_size/2)]
            #patch = gray[int(y - patch_size // 2):int(y + patch_size // 2 + 1), int(x - patch_size // 2):int(x + patch_size // 2 + 1)]
        # patch = cv2.GaussianBlur(patch, (3, 3), 0)

        else:
            print("Patch is None")
            continue

        # Apply Gaussian blur to the patch
        patch = cv2.GaussianBlur(patch, (3, 3), blur_sigma)

        patch_subsampled = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA)
        feature_vector = patch_subsampled.reshape(64)
        feature_vector_std = (feature_vector - feature_vector.mean()) / (feature_vector.std() + 1e-10)

        # Sub-sample the blurred patch
        #patch = patch[::patch_size // sub_sample_size, ::patch_size // sub_sample_size]

        # Standardize the patch
        #patch = (patch - np.mean(patch)) / np.std(patch)

        # Flatten the patch into a feature vector
        #feature_vector = patch.flatten()

        #print(feature_vector_std.shape)

        # Append the feature vector to the list of feature descriptors
        feature_descriptors.append(feature_vector_std)

    return feature_descriptors

def match_features(features1, features2, corners1, corners2, threshold=1):
    matched_coords = []
    for i in range(len(features1)):
        SSD = []
        for j in range(len(features2)):
            ssd_val = np.linalg.norm((features1[i]-features2[j]))**2
            SSD.append(ssd_val)

        sorted_SSD_index = np.argsort(SSD)
        if (SSD[sorted_SSD_index[0]] / SSD[sorted_SSD_index[1]]) < threshold:
            matched_coords.append((corners1[i], corners2[sorted_SSD_index[0]]))

    print(f"Number of matched co-ordinates: {len(matched_coords)}")

    if len(matched_coords) < 30:
        print('\nNot enough matches!\n')
        quit()

    return np.array(matched_coords)

def visualizeMatchedFeatures(image1, image2, matched_coords):
    """
        Takes two images and list of co-ordiates of matched features in the images
        and gives a visulaization of the mapping of these features across the images.
    """
    img1 = image1.copy()
    img2 = image2.copy()
    print("\nVisualizing the mapping of matched features of the two images.")

    # images has to be resized into same size (height needs to be same for concatenation)
    image_sizes = [img1.shape, img2.shape]
    target_size = np.max(image_sizes, axis=0)
    if img1.shape != list(target_size):
        print("\nResizing image 1 for proper visualization.")
        img1 = cv2.resize(img1, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    if img2.shape != list(target_size):
        print("\nResizing image 2 for proper visualization.")
        img2 = cv2.resize(img2, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    
    concatenated_image = np.concatenate((img1, img2), axis=1)
    corners1, corners2 = matched_coords[:, 0].astype(int).copy(), matched_coords[:, 1].astype(int).copy()
    corners2[:,0] += img1.shape[1]
    
    for coord1, coord2 in zip(corners1, corners2):
        cv2.line(concatenated_image, (coord1[0], coord1[1]), (coord2[0], coord2[1]), (0, 255, 255), 1)
        cv2.circle(concatenated_image, (coord1[0], coord1[1]), 3, (0,0,255), 1)
        cv2.circle(concatenated_image, (coord2[0], coord2[1]), 3, (0,255,0), 1)
    
    cv2.imwrite('matching_sample.png', concatenated_image)

def RANSAC(matched_coords,accuracy=0.9, threshold=5):
    """
        Takes the list of matched co-ordinates of features in two images and
        returns the homography matrix between the two images.
    """
    print("\nEstimating homography between the two images using RANSAC.")
    matched_coords = matched_coords.astype(int)
    num_matches = len(matched_coords)
    #num_iterations = int(np.log(1-accuracy)/np.log(1-(1-threshold/num_matches)**4))
    num_iterations = 3000
    print(f"Number of iterations: {num_iterations}")
    best_inliers = []
    best_homography = None
    for i in range(num_iterations):
        # Randomly select 4 points
        random_indices = np.random.choice(num_matches, size=4, replace=False)
        random_coords = matched_coords[random_indices]
        # Compute homography
        homography = cv2.getPerspectiveTransform(np.float32(random_coords[:, 0]), np.float32(random_coords[:, 1]))
        # Find inliers
        inliers = []
        for j in range(num_matches):
            if np.linalg.norm(np.dot(homography, np.array([matched_coords[j][0][0], matched_coords[j][0][1], 1])) - np.array([matched_coords[j][1][0], matched_coords[j][1][1], 1])) < threshold:
                inliers.append(matched_coords[j])
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_homography = homography
    print(f"Number of inliers: {len(best_inliers)}")
    return best_homography

#Write a function to warp the image using the homography matrix. You can use OpenCV’s warpPerspective function.
def warpandblend(image1, image2, homography):
    """
        Takes two images and homography matrix between them and returns the warped
        image of the first image.
    """
    print("\nWarping the first image using the homography matrix.")
    warped_image = cv2.warpPerspective(image1, homography, (image1.shape[1]+image2.shape[1], image1.shape[0]))
    warped_image[0:image2.shape[0], 0:image2.shape[1]] = image2
    return warped_image
  
# def warp_image(image, homography):
#     # Warp the image using the homography
#     warped_image = cv2.warpPerspective(image, homography, (image.shape[1], image.shape[0]))

#     return warped_image

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
    
    # filename1 = '/Users/rohin/Desktop/WPI_Fall_ 2022_Academics/RBE_549/P1/YourDirectoryID_p1/Phase1/Data/Train/Set1/1.jpg'
    # img1 = cv2.imread(filename1)
    
    filename1 = '/Users/rohin/Desktop/WPI_Fall_ 2022_Academics/RBE_549/P1/YourDirectoryID_p1/Phase1/Data/Train/Set3/2.jpg'
    img1 = cv2.imread(filename1)
    
    filename2 = '/Users/rohin/Desktop/WPI_Fall_ 2022_Academics/RBE_549/P1/YourDirectoryID_p1/Phase1/Data/Train/Set3/3.jpg'
    img2 = cv2.imread(filename2)
    
    #images = [img1,img2]
    
    #cornerDetection(filename1)

    print(np.shape(img1))

    corners_1, corner_coords_1, output_image_1 = extract_corners(filename1, use_harris=True)

    best_corners_1 = ANMS(img1, corners_1, 700)

    feature_descriptors_1 = feature_descriptor(img1, best_corners_1,patch_size=41, blur_sigma=0)

    corners_2, corner_coords_2, output_image_2 = extract_corners(filename2, use_harris=True)

    best_corners_2 = ANMS(img2, corners_2, 700)

    feature_descriptors_2 = feature_descriptor(img2, best_corners_2,patch_size=41, blur_sigma=0)

    matching_indices = match_features(feature_descriptors_1, feature_descriptors_2, best_corners_1, best_corners_2, threshold=1)

    print(matching_indices)

    visualizeMatchedFeatures(img1, img2, np.array(matching_indices))

    homography = RANSAC(matching_indices,accuracy=0.9, threshold=5)

    print(homography)

    warped_image = warpandblend(img1, img2, homography)

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
