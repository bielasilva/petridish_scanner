#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import os

# Utility function to vectorize integer conversion
f_int = np.vectorize(int)

# Class to handle image loading, saving, and preprocessing (masks and contours)
class ImageProcessor:
    def __init__(self, image_file):
        """
        Initialize ImageProcessor with the file name of the image.
        
        :param image_file: Path to the image file to be processed
        """
        self.image_file = image_file
        self.img = cv2.imread(image_file)  # Load the image
        self.mask = None  # Mask will be generated later

    def save_image(self, output_file, img):
        """
        Save the processed image to the specified output file.
        
        :param output_file: The file path where the image should be saved
        :param img: The image data to save
        """
        cv2.imwrite(output_file, img)

    def get_mask(self, style="canny"):
        """
        Apply different types of image processing masks based on the specified style.
        
        :param style: The type of mask to apply (canny, aruco, contoured, hsv_transform)
        :return: The resulting mask
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        if style == "canny":
            # Use Canny edge detection with dilation and erosion for a clean edge mask
            edged = cv2.Canny(gray, 20, 10)
            edged = cv2.dilate(edged, None, iterations=1)
            self.mask = cv2.erode(edged, None, iterations=1)
        
        elif style == "aruco":
            # Simple binary threshold for detecting ArUco markers
            self.mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
        
        elif style == "contoured":
            # Adaptive thresholding for contour detection
            self.mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 241, 2)
            self.mask = cv2.dilate(self.mask, None, iterations=1)
            self.mask = cv2.erode(self.mask, None, iterations=1)
        
        elif style == "hsv_transform":
            # Use HSV color space for more complex masking (e.g., color-based masking)
            img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(img_hsv, np.array([0, 50, 50]), np.array([179, 255, 255]))
            img_hsv_modify = cv2.bitwise_and(self.img, self.img, mask=mask)
            img_mask_gray = cv2.cvtColor(img_hsv_modify, cv2.COLOR_BGR2GRAY)
            self.mask = cv2.threshold(img_mask_gray, 0, 255, cv2.THRESH_BINARY)[1]
        
        return self.mask

    def get_contours(self):
        """
        Detect contours in the current mask and filter out smaller areas.
        
        :return: A list of filtered contours
        """
        # Find contours in the mask
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter contours based on area (ignore small areas)
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 10**5]
        return filtered_contours

# Class to handle ArUco marker detection and pixel-to-cm ratio calculation
class ArUcoDetector:
    def __init__(self, mask):
        """
        Initialize the ArUcoDetector with a mask where ArUco markers can be detected.
        
        :param mask: Binary mask of the image where ArUco markers are to be detected
        """
        self.mask = mask
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)  # ArUco dictionary definition
        self.aruco_params = cv2.aruco.DetectorParameters()  # ArUco marker detection parameters

    def detect_aruco(self):
        """
        Detect ArUco markers in the provided mask image.
        
        :return: A dictionary of detected ArUco corners, keyed by marker ID
        """
        # Detect ArUco markers in the image using the pre-defined dictionary
        corners, ids, _ = cv2.aruco.detectMarkers(self.mask, self.aruco_dict, parameters=self.aruco_params)
        aruco_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # List of ArUco IDs to detect
        aruco_corners = {}

        if ids is not None:
            # Loop over expected IDs and extract the corresponding corners
            for i in aruco_ids:
                try:
                    aruco_idx = np.where(ids == i)[0][0]
                    aruco_corners[str(i)] = corners[aruco_idx]
                except IndexError:
                    continue  # Skip if the marker ID is not found
        return aruco_corners

    def get_pixel_cm_ratio(self, aruco_ref):
        """
        Calculate the pixel-to-centimeter ratio using detected ArUco markers.
        
        :param aruco_ref: Detected ArUco markers in the reference image
        :return: The pixel-to-cm ratio for the image
        """
        # Separate detected ArUco markers into two groups
        aruco1_ids = ["1", "2", "3", "4"]
        aruco2_ids = ["5", "6", "7", "8", "9", "10", "11", "12"]

        # Extract corners for ArUco groups
        aruco1_corners = [np.array(aruco_ref[f"{id}"]) for id in aruco1_ids if f"{id}" in aruco_ref]
        aruco2_corners = [np.array(aruco_ref[f"{id}"]) for id in aruco2_ids if f"{id}" in aruco_ref]

        # Calculate perimeters of ArUco markers in both groups
        aruco1_perimeters = [cv2.arcLength(corners, True) for corners in aruco1_corners]
        aruco2_perimeters = [cv2.arcLength(corners, True) for corners in aruco2_corners]

        # Compute average perimeters for more robust ratio calculation
        aruco1_perimeter = np.mean([p for p in aruco1_perimeters if p > np.mean(aruco1_perimeters)])
        aruco2_perimeter = np.mean([p for p in aruco2_perimeters if p > np.mean(aruco2_perimeters)])

        # Compute pixel-to-cm ratio as the average between both groups of ArUco markers
        return np.mean([(aruco1_perimeter / 4), (aruco2_perimeter / 8)])

# Class to handle perspective correction between sample and reference images
class PerspectiveCorrector:
    def __init__(self, img_sample, aruco_sample, img_ref, aruco_ref):
        """
        Initialize the PerspectiveCorrector with the sample and reference images and detected ArUco markers.
        
        :param img_sample: The sample image to be corrected
        :param aruco_sample: Detected ArUco markers in the sample image
        :param img_ref: The reference image used for correction
        :param aruco_ref: Detected ArUco markers in the reference image
        """
        self.img_sample = img_sample
        self.aruco_sample = aruco_sample
        self.img_ref = img_ref
        self.aruco_ref = aruco_ref

    def correct_perspective(self):
        """
        Correct the perspective of the sample image based on the detected ArUco markers.
        
        :return: The perspective-corrected sample image
        """
        # Find matching ArUco markers between sample and reference images
        detected_sample = list(self.aruco_sample.keys())
        filtered_ref = {k: v for k, v in self.aruco_ref.items() if k in detected_sample}

        # Get corresponding corners of detected ArUco markers
        ref_corners = np.array(list(filtered_ref.values())).reshape(-1, 1, 2)
        sample_corners = np.array(list(self.aruco_sample.values())).reshape(-1, 1, 2)

        # Compute the homography matrix to warp the sample image to match the reference
        M, _ = cv2.findHomography(sample_corners, ref_corners, cv2.RANSAC)

        # Apply perspective transformation (warping) to the sample image
        corrected_img = cv2.warpPerspective(self.img_sample, M, (self.img_ref.shape[1], self.img_ref.shape[0]))

        return corrected_img

# Class to handle measurement and drawing information on images
class ObjectAnalyzer:
    def __init__(self, img, contours, pixel_cm_ratio):
        """
        Initialize the ObjectAnalyzer with the processed image, contours, and pixel-to-cm ratio.
        
        :param img: The image in which objects will be analyzed
        :param contours: List of contours detected in the image
        :param pixel_cm_ratio: The ratio to convert pixels to centimeters
        """
        self.img = img
        self.contours = contours
        self.pixel_cm_ratio = pixel_cm_ratio

    def get_biggest_rect(self):
        """
        Find the contour with the largest perimeter (assuming it represents the main object).
        
        :return: The minimum-area rectangle that encloses the largest contour
        """
        max_perim = 0
        max_rect = None

        # Loop over contours to find the largest by perimeter
        for cnt in self.contours:
            rect = cv2.minAreaRect(cnt)  # Find minimum area rectangle
            box = f_int(cv2.boxPoints(rect))  # Convert rectangle points to integers
            perim_len = cv2.arcLength(box, True)  # Compute perimeter length of the rectangle
            if perim_len > max_perim:
                max_perim = perim_len
                max_rect = rect  # Store the largest rectangle

        return max_rect

    def get_object_measurements(self, rect):
        """
        Calculate the width and height of the object in centimeters.
        
        :param rect: The minimum-area rectangle enclosing the object
        :return: Object width and height in centimeters
        """
        (x, y), (w, h), angle = rect
        object_width = w / self.pixel_cm_ratio
        object_height = h / self.pixel_cm_ratio
        return object_width, object_height

    def draw_object_info(self, print_box=False):
        """
        Draw information about the object's dimensions and optionally its bounding box on the image.
        
        :param print_box: Whether to draw the bounding box of the object on the image
        :return: The image with the drawn object information
        """
        # Get the largest rectangle (assumed to be the main object)
        rect_object = self.get_biggest_rect()
        object_width, object_height = self.get_object_measurements(rect_object)

        # Extract rectangle properties
        (x, y), (w, h), angle = rect_object

        if print_box:
            # Draw the bounding box around the object
            box = f_int(cv2.boxPoints(rect_object))
            cv2.drawContours(self.img, [box], 0, (0, 255, 0), 10)
            # Display object dimensions (width and height) on the image
            cv2.putText(self.img, f"Width: {round(object_width, 1)} cm", (int(x/1.5), int(y*1.5 + 200)),
                        cv2.FONT_HERSHEY_PLAIN, 25, (100, 200, 0), 20)
            cv2.putText(self.img, f"Height: {round(object_height, 1)} cm", (int(x/1.5), int(y*1.5 + 500)),
                        cv2.FONT_HERSHEY_PLAIN, 25, (100, 200, 0), 20)

        return self.img

# Main handler class to manage the entire image processing workflow
class ImageProcessingPipeline:
    def __init__(self, ref_file, sample_files):
        """
        Initialize the ImageProcessingPipeline with the reference and sample file paths.
        
        :param ref_file: Path to the reference image file
        :param sample_files: List of paths to the sample image files
        """
        self.ref_file = ref_file
        self.sample_files = sample_files
        self.pixel_cm_ratio = None

    def run(self):
        """
        Execute the complete image processing workflow:
        - Load and process images
        - Detect ArUco markers
        - Correct perspective of the sample images
        - Analyze objects and draw information
        - Save the final processed images
        """
        # Step 1: Load and process the reference image
        ref_processor = ImageProcessor(self.ref_file)
        mask_aruco_ref = ref_processor.get_mask(style="aruco")

        # Detect ArUco markers in the reference image
        aruco_detector_ref = ArUcoDetector(mask_aruco_ref)
        aruco_ref = aruco_detector_ref.detect_aruco()

        # Calculate pixel-to-cm ratio using reference ArUco markers
        self.pixel_cm_ratio = aruco_detector_ref.get_pixel_cm_ratio(aruco_ref)

        # Step 2: Process each sample image
        for sample_file in self.sample_files:
            print(f"Processing {sample_file}...")

            # Step 2.1: Load the sample image and generate its mask
            sample_processor = ImageProcessor(sample_file)
            mask_aruco_sample = sample_processor.get_mask(style="aruco")

            # Step 2.2: Detect ArUco markers in the sample image
            aruco_detector_sample = ArUcoDetector(mask_aruco_sample)
            aruco_sample = aruco_detector_sample.detect_aruco()

            # Step 2.3: Correct the perspective of the sample image
            perspective_corrector = PerspectiveCorrector(sample_processor.img, aruco_sample, ref_processor.img, aruco_ref)
            corrected_img = perspective_corrector.correct_perspective()

            # Step 2.4: Generate the mask and detect contours in the corrected image
            corrected_processor = ImageProcessor(sample_file)
            corrected_processor.img = corrected_img
            mask_corrected = corrected_processor.get_mask(style="hsv_transform")
            contours_corrected = corrected_processor.get_contours()

            # Step 2.5: Analyze the object and draw the bounding box and measurements
            object_analyzer = ObjectAnalyzer(corrected_img, contours_corrected, self.pixel_cm_ratio)
            labeled_img = object_analyzer.draw_object_info(print_box=True)

            # Step 2.6: Save the processed image
            output_file = f"{os.path.splitext(sample_file)[0]}_processed.jpg"
            sample_processor.save_image(output_file, labeled_img)

            print(f"Saved processed image to {output_file}")


# Function to parse command-line arguments
def parse_args():
    """
    Parse command-line arguments using argparse.
    
    :return: Parsed arguments containing reference file and sample file paths
    """
    parser = argparse.ArgumentParser(description="Image processing pipeline with ArUco markers and object detection.")
    parser.add_argument('-s', '--samples', nargs='+', required=True, help="Paths to one or more sample image files")
    parser.add_argument('-r', '--reference', default="Reference_sheet.png", help="Path to the reference image file")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Get reference and sample file paths from the command-line arguments
    ref_file = args.reference
    sample_files = args.samples
    
    # Run the image processing pipeline with provided file paths
    pipeline = ImageProcessingPipeline(ref_file, sample_files)
    pipeline.run()