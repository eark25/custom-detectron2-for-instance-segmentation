import numpy as np

# Create an image
img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
print(img)
# Define the coordinates array
coordinates = np.array([[1, 1], [1, 1], [0, 2]])
print(coordinates)
# Get the pixel values at the specified coordinates
pixel_values = img[coordinates[:,0], coordinates[:,1]]

# Print the pixel values
print(pixel_values)

print("Most frequent value in above array")
y = np.bincount(pixel_values)
print(y)
maximum = max(y)
  
for i in range(len(y)):
    if y[i] == maximum:
        print(i, end=" ")