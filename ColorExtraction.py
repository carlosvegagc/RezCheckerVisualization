import cv2
import csv
import numpy as np

index_2_extract = [1, 2, 3, 4, 6, 11, 12, 17, 18, 19, 20, 21,22,23,24,25,26,27,28,29,30,31, 32, 33, 34, 35, 37, 38, 39, 40]

# Print number index to extract
print(len(index_2_extract))

def extract_lab_colors(image_path):

    # Get the image name
    image_name = image_path.split('\\')[-1].split('.')[0]

    # Load the image
    image = cv2.imread(image_path)

    print(image.shape)

    # print image type
    print(type(image))
    print(image.dtype)

    image_int = image.astype(np.int16)


    # Identify grayscale values in the original image. 
    # If all three channels have the same value, then it's grayscale.
    threshold = 30
    grayscale_mask = ( np.abs(image_int[:, :, 0]  - image_int[:, :, 1]) < threshold ) & ( np.abs(image_int[:, :, 0] - image_int[:, :, 2]) < threshold ) & ( np.abs(image_int[:, :, 1] - image_int[:, :, 2]) < threshold )

    # Create an output image filled with zeros (black)
    output = np.ones_like(image)

    # Set grayscale values in the output image
    output[grayscale_mask] = image[grayscale_mask]

    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    
    gray[gray < 40] = 0
    gray[gray > 40] = 255
    
    # Show the image
    cv2.imshow('image', gray)

    cv2.waitKey(0)

    # Threshold the image to get the target
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    #thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)

    # If close pixels are true, connect them together
    kernel = np.ones((8, 8), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Show the image
    cv2.imshow('image', thresh)
    cv2.waitKey(0)

    # Detect the corner dots using contour detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#     # Plot the contours on the image
#     cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

#   # Wait for a key to be pressed
#     cv2.waitKey(0)
 
    # Filter out small contours that are not the dots
    threshold_relation_l = 0.7
    threshold_relation_h = 1.3
    min_radius = 2
    max_radius = 20000
    centroids = []

    for contour in contours:

        # Get the center and radius of the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # Check if the radius falls within the specified range
        if min_radius <= radius <= max_radius:

            # Further checks can be added here, like circularity criteria
            circularity = 4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)

            if circularity > 0.85 :
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Get width and height of bounding box
                x, y, w, h = cv2.boundingRect(contour)
                if w / h > threshold_relation_l and w / h < threshold_relation_h:  # Adjust the threshold as necessary
                    centroids.append((cX, cY))

            
    print(len(centroids))

   
    # Order the centroids from top-left to bottom-right
    centroids = np.array(centroids)
    centroids = centroids[np.argsort(centroids[:, 1])]
    centroids[:2] = centroids[:2][np.argsort(centroids[:2, 0])]
    centroids[2:] = centroids[2:][np.argsort(centroids[2:, 0])]
    centroids = [tuple(c) for c in centroids]
    
    # Plot the centroids and its number on the image
    for index, c in enumerate(centroids):
        cv2.circle(image, c, 5, (0, 0, 255), -1)
        cv2.putText(image, str(centroids.index(c)), c, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
   # Show the image
    cv2.imshow('image', image)

    # Wait for a key to be pressed
    cv2.waitKey(0)

    # Sort the centroids to get the bounding box of the target
    top_left, top_right,bottom_left, bottom_right = centroids


    # Compute grid distances
    num_rows, num_cols = 7,6  # Adjust if different
    grid_width = (top_right[0] - top_left[0]) / (num_cols - 1)
    grid_height = (bottom_left[1] - top_left[1]) / (num_rows -1)

    # Plot the grid
    for i in range(num_rows ):
        y = int(top_left[1] + i * grid_height)
        cv2.line(image, (0, y), (image.shape[1], y), (255, 0, 0), 1)

    for j in range(num_cols ):
        x = int(top_left[0] + j * grid_width)
        cv2.line(image, (x, 0), (x, image.shape[0]), (255, 0, 0), 1)

    
    # Convert the image to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)
    # Extract the colors from the grid positions
    colors = []
    coordinates = []
    color_variability = []
    for i in range(num_rows):
        for j in range(num_cols):
            x = int(top_left[0] + j * grid_width )  # center x-coordinate of the grid
            y = int(top_left[1] + i * grid_height )  # center y-coordinate of the grid
            grid_region = lab_image[y-int(grid_height/5):y+int(grid_height/5), x-int(grid_width/5):x+int(grid_width/5)]
            color = grid_region.mean(axis=0).mean(axis=0)
            variability = grid_region.std(axis=0).std(axis=0)
            colors.append(tuple(map(int, color)))
            coordinates.append((x, y))
            color_variability.append(tuple(map(float, variability)))
    
    # Print the colors of the grid on the image
    for index, coord in enumerate(coordinates):
        cv2.putText(image, str(index), coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Force the program to wait here until a key is pressed
    cv2.waitKey(0)


    # Save to CSV
    with open('colors.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['X', 'Y', 'L', 'a', 'b', 'Var_L', 'Var_a', 'Var_b'])  # column headers
        for index in index_2_extract:
            coord = coordinates[index]
            color = colors[index]
            variability = color_variability[index]
            writer.writerow([coord[0], coord[1], color[0], color[1], color[2], variability[0], variability[1], variability[2]])


    # Show the image
    cv2.imshow('image', image)

    # Save the cv2 image
    cv2.imwrite('Colors_extracted_'+image_name +'.jpg', image)
    

    # Force the program to wait here until a key is pressed
    cv2.waitKey(0)



if __name__ == "__main__":
    image_path = './RezChecker_2.jpeg'
    extract_lab_colors(image_path)