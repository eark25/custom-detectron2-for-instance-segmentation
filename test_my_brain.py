# import numpy as np

# # Create an image
# img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
# print(img)
# # Define the coordinates array
# coordinates = np.array([[1, 1], [1, 1], [0, 2]])
# print(coordinates)
# # Get the pixel values at the specified coordinates
# pixel_values = img[coordinates[:,0], coordinates[:,1]]

# # Print the pixel values
# print(pixel_values)

# print("Most frequent value in above array")
# y = np.bincount(pixel_values)
# print(y)
# maximum = max(y)
  
# for i in range(len(y)):
#     if y[i] == maximum:
#         print(i, end=" ")


import numpy as np
import cv2
# # Create a blank image
# img = np.zeros((100, 100), dtype=np.uint8)

# # Define the polygon coordinates
# polygon_coordinates = np.array([[10,10], [30,10], [30,30], [10,30], [10, 10]], dtype=np.int32)

# # Draw the polygon on the image
# cv2.fillPoly(img, [polygon_coordinates], (255))
# cv2.imwrite('check.png', img)

# # Get the indices of non-zero elements in the image
# indices = np.transpose(np.nonzero(img))
# print(indices)
# print(indices.shape)

# img = np.zeros((100, 100), dtype=np.uint8)
# cv2.fillPoly(img, [indices], (255))
# cv2.imwrite('recheck.png', img)

# # Get the pixel values of each pixel that the polygon coordinates are on
# pixel_values = img[indices[:, 0], indices[:, 1]]

# # Print the pixel values
# print(pixel_values)
# print(pixel_values.shape)

# point coordinates
# img = np.zeros((3000, 4000), dtype=np.uint8)
# h, w = img.shape
# print(w, h)
# x, y = (1796, 1346)
# x, y = (4000, 3000)
# print(x, y)

# # horizontal
# if 0 <= x < w//4:
#     print("The point {} belongs to left one fourth: 0 to {}".format(x, w//4))
# elif w//4 <= x < w//2:
#     print("The point {} belongs to middle fourths: {} to {}".format(x, w//4, w//2))
# elif w//2 <= x < 3*w//4:
#     print("The point {} belongs to middle fourths: {} to {}".format(x, w//2, 3*w//4))
# else:
#     print("The point {} belongs to right one fourth: {} to {}".format(x, 3*w//4, w))

# # vertical
# if 0 <= y < h//4:
#     print("The point {} belongs to top fourth: 0 to {}".format(y, h//4))
# elif h//4 <= y < h//2:
#     print("The point {} belif (1 and 2) in top_parts or 1 in top_parts:
#         print('Top fourths')
#     elif (3 and 4) in top_parts or 4 in top_parts:
#         print('Bottom fourths')
#     else:
#         print('Middle fourths')ongs to middle fourths: {} to {}".format(y, h//4, h//2))
# elif h//2 <= y < 3*h//4:
#     print("The point {} belongs to middle fourths: {} to {}".format(y, h//2, 3*h//4))
# else:
#     print("The point {} belongs to bottom fourth: {} to {}".format(y, 3*h//4, h))

# top_parts = [2, 1]
# print(1 in top_parts and 2 in top_parts or 1 in top_parts)
# if 1 in top_parts and 2 in top_parts or 1 in top_parts:
#     print('Top fourths')
# elif 3 in top_parts and 4 in top_parts or 4 in top_parts:
#     print('Bottom fourths')
# else:
#     print('Middle fourths')
# if top_parts[0] == 1:
#     print('Left one fourth')
# elif top_parts[0] == 4:
#     print('Right one fourth')
# else:
#     print('Middle fourths')


import numpy as np

arr = np.array([[1,2,3,4],
                [5,3,6,8]])

flipped_arr = np.transpose(arr)[:,::-1]

print(flipped_arr)