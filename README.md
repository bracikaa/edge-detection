# Canny Edge Detection Algorithm

Canny edge detection is a complex, multilayered algorithm that detects edges. It was named after its creator John Canny, and it is widely used in Computer Vision, even in computationally intensive scenarios, due to its accuracy and effectiveness. During the whole process algorithm takes some additional steps that suppress noise and minimize false detections, in order to get the best edge-detection results.  The steps involved are:

Denoising using a Gaussian Filter

Gradient calculation

NMS (non-maximum suppression) applied on the edges

Applying the double threshold

Tracking edges

Code that accompanies the following article: