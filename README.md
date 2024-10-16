# PetriDish Scanner and Measuring Tool

This project is an image processing pipeline that uses ArUco markers to detect and analyze objects in images. The pipeline can correct perspective distortions, calculate object dimensions, and draw bounding boxes around detected objects. The project is designed to process multiple images in batch mode and save the processed output.

## Features

* Perspective Correction: Uses ArUco markers to correct the perspective of an image based on a reference image.
* Object Detection: Detects objects in the image using contours.
* Dimension Measurement: Computes the width and height of detected objects in real-world units (e.g., centimeters).
* Batch Processing: Supports multiple sample images at once.
* Automatic Mask Generation: Uses different masking techniques (Canny, ArUco, HSV, etc.) to detect features and contours.

## Requirements

This project requires Python 3 and the following Python libraries:

* opencv-python
* numpy
* argparse

You can install these dependencies using pip:

```bash
pip install opencv-python numpy
```

## Getting Started

### Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/aruco-image-processing.git
cd aruco-image-processing
```

### Running the Pipeline

The pipeline can be executed through the command line. You can provide multiple sample images as input, along with a reference image that contains the ArUco markers used for perspective correction.

#### Usage

```bash
python scanner.py --reference <path_to_reference_image> --samples <path_to_sample_image1> <path_to_sample_image2> ...
```

#### Arguments

* --reference (-r): Path to the reference image containing ArUco markers (default: Reference_sheet.png).
* --samples (-s): Paths to one or more sample image files that you want to process.

#### Example

```bash
python scanner.py --reference Reference_sheet.png --samples sample1.jpg sample2.jpg sample3.jpg
```

This will process each of the provided sample images and save the processed output as <sample_name>_processed.jpg.

#### Processing a Directory of Images

If you have a directory full of images and you’d like to process all the images in it, you can use a script to pass all files in a directory to the pipeline:

```bash
python script_name.py --reference Reference_sheet.png --samples images/*.jpg
```

#### Output

For each input image, the pipeline:

1. Detects ArUco markers in the image for perspective correction.
2. Corrects the perspective of the image based on the reference image.
3. Detects objects (e.g., rectangles) and calculates their dimensions.
4. Draws bounding boxes around detected objects and overlays the dimensions.
5. Saves the processed image as <sample_name>_processed.jpg.

## Project Structure

The project directory is organized as follows:

```plaintext
└── petridish_scanner/
    ├── scanner.py              # Main Python script for running the pipeline
    ├── marker_generator.py         # Python script for generating ArUco marker sheets for reference
    ├── README.md                   # Project documentation
    ├── Reference_sheet.png         # Example reference image containing ArUco markers
    └── requirements.txt            # Project dependencies
```
