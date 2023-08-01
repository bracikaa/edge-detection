import cv2
import numpy as np

def generate_gaussian_kernel(kernel_size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2))
        * np.exp(
            -((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2)
            / (2 * sigma**2)
        ),
        (kernel_size, kernel_size),
    )
    kernel /= np.sum(kernel)
    return kernel

def gradient_calculation(input_image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradient_x = cv2.filter2D(input_image, cv2.CV_32F, sobel_x)
    gradient_y = cv2.filter2D(input_image, cv2.CV_32F, sobel_y)

    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    gradient_magnitude = gradient_magnitude.astype('uint8')

    gradient_direction = np.rad2deg(np.arctan2(gradient_y, gradient_x))
    gradient_direction= gradient_direction.astype('uint8')
    cv2.imshow('direction', gradient_direction)

    return gradient_magnitude, gradient_direction

def non_maximum_suppression(magnitude, direction):
    shape = magnitude.shape
    suppressed_image = np.zeros(shape)

    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            if (0 <= direction[i, j] < 22.5) or (157.5 <= direction[i, j] <= 180):
                value_to_compare = max(magnitude[i, j - 1], magnitude[i, j + 1])
            elif (22.5 <= direction[i, j] < 67.5):
                value_to_compare = max(magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])
            elif (67.5 <= direction[i, j] < 112.5):
                value_to_compare = max(magnitude[i - 1, j], magnitude[i + 1, j])
            else:
                value_to_compare = max(magnitude[i + 1, j - 1], magnitude[i - 1, j + 1])
            
            if magnitude[i, j] >= value_to_compare:
                suppressed_image[i, j] = magnitude[i, j]
    suppressed_image = np.multiply(suppressed_image, 255.0 / suppressed_image.max())
    return suppressed_image

def double_threshold_and_edge_tracking(image, low_threshold, high_threshold):
    weak = 50
    strong = 255
    size = image.shape
    result = np.zeros(size)
    weak_x, weak_y = np.where((image > low_threshold) & (image <= high_threshold))
    strong_x, strong_y = np.where(image >= high_threshold)
    result[strong_x, strong_y] = strong
    result[weak_x, weak_y] = weak
    dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
    dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
    size = image.shape
    
    while len(strong_x):
        x = strong_x[0]
        y = strong_y[0]
        strong_x = np.delete(strong_x, 0)
        strong_y = np.delete(strong_y, 0)
        for direction in range(len(dx)):
            new_x = x + dx[direction]
            new_y = y + dy[direction]
            if((new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (result[new_x, new_y]  == weak)):
                result[new_x, new_y] = strong
                np.append(strong_x, new_x)
                np.append(strong_y, new_y)
    result[result != strong] = 0
    return result

if __name__ == "__main__":
    img = cv2.imread("./Lenna.png", cv2.IMREAD_GRAYSCALE)
    gauss_kernel = generate_gaussian_kernel(5, 1.2)
    blurred_image = cv2.filter2D(img, -1, gauss_kernel)
    magnitude, direction = gradient_calculation(blurred_image)
    edge_map = non_maximum_suppression(magnitude, direction)
    low_threshold = [0.05, 0.10, 0.15, 0.20, 0.25]
    high_threshold = [0.35, 0.40, 0.45, 0.50, 0.55]
    result = double_threshold_and_edge_tracking(edge_map, low_threshold=0, high_threshold=70)
    canny_algorithm = cv2.Canny(img,100,200)

    cv2.imshow("Image", img)
    cv2.imshow("Blurred Image", blurred_image)
    cv2.imshow('Edge Map', edge_map)
    cv2.imshow('Result', result)
    cv2.imshow('Canny', canny_algorithm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

