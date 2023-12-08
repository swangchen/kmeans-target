import cv2
import numpy as np

# Load the cropped image and the original image
cropped_image = cv2.imread('cropped_center_region.jpg')
original_image = cv2.imread('2.jpg')

# Ensure both images have the same size
cropped_image = cv2.resize(cropped_image, (original_image.shape[1], original_image.shape[0]))

# Convert the cropped image to grayscale
gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

# Create a binary mask by thresholding the grayscale image
_, mask = cv2.threshold(gray_cropped, 1, 255, cv2.THRESH_BINARY)

# Invert the mask
mask_inv = cv2.bitwise_not(mask)

# Convert the cropped image to a 4-channel image
cropped_alpha = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)

# Set the alpha channel of the cropped image using the inverted mask
cropped_alpha[:, :, 3] = mask_inv

# Set the alpha channel of the original image using the mask
original_alpha = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
original_alpha[:, :, 3] = mask

# Subtract the original image from the cropped image with alpha channels
result_alpha = cv2.subtract(cropped_alpha, original_alpha)

# Extract the result without the alpha channel
result = result_alpha[:, :, :3]

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
