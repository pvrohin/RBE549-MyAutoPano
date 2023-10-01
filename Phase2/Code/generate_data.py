import cv2
import numpy as np
import random

#Read an image from the Data/train folder
img = cv2.imread('/Users/rohin/Desktop/WPI_Fall_ 2022_Academics/RBE_549/P1/YourDirectoryID_p1/Phase2/Data/Train/2.jpg')

#Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Obtain a random patch from the image such that all the pixels in the patch will lie within the image after warping the random extracted patch. 
def extract_random_patch_with_perturbation(IA, MP, NP, rho):
    # Image dimensions
    M, N = IA.shape
    
    # Calculate maximum allowable shift
    SHmax = min(rho, M - MP)
    SVmax = min(rho, N - NP)

    # Randomly select starting position within allowable range
    SH = random.randint(0, SHmax)
    SV = random.randint(0, SVmax)

    # Calculate ending position
    patch_start = (SH, SV)
    patch_end = (SH + MP, SV + NP)

    # Check if the entire patch remains within the image after perturbation
    if (patch_start[0] >= 0 and patch_start[1] >= 0 and
        patch_end[0] <= M and patch_end[1] <= N):
        
        # Extract the patch PA from IA
        PA = IA[patch_start[0]:patch_end[0], patch_start[1]:patch_end[1]]
        
        return PA
    else:
        # Patch is not entirely within the image, return None or handle the case as needed.
        return None

# Example usage:
# Replace this with your actual image and desired parameters
IA = gray  # Sample image
MP, NP = 128, 128  # Patch size
rho = 10  # Maximum perturbation

random_patch = extract_random_patch_with_perturbation(IA, MP, NP, rho)
if random_patch is not None:
    print("Random Patch Extracted:")
    print(random_patch)
    #display the patch
    cv2.imshow('Random Patch', random_patch)
    cv2.waitKey(0)
else:
    print("Patch extraction failed. Adjust parameters or handle this case accordingly.")


