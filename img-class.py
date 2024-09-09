import cv2
import os
import concurrent.futures
import numpy as np

def load_image(image_path):
    """
    Load an image from a file path using OpenCV.

    Args:
        image_path (str): The path to the image file.

    Returns:
        image (numpy.ndarray): The loaded image.

    Raises:
        FileNotFoundError: If the image file does not exist.
        Exception: If the image file cannot be loaded.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Failed to load image file: {image_path}")

    return image

def apply_image_processing(image, techniques):
    """
    Apply image processing techniques to the loaded image.

    Args:
        image (numpy.ndarray): The loaded image.
        techniques (list): A list of image processing techniques to apply.

    Returns:
        processed_images (dict): A dictionary of processed images.
    """
    processed_images = {}

    if 'grayscale' in techniques:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_images['grayscale'] = gray_image

    if 'resize' in techniques:
        resized_image = cv2.resize(image, (800, 600))
        processed_images['resized'] = resized_image

    if 'flip' in techniques:
        flipped_image = cv2.flip(image, 1)
        processed_images['flipped'] = flipped_image

    if 'edge_detection' in techniques:
        edges = cv2.Canny(image, 100, 200)
        processed_images['edge_detection'] = edges

    if 'object_recognition' in techniques:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        processed_images['object_recognition'] = image

    return processed_images

def display_images(original_image, processed_images):
    """
    Display the original and processed images using OpenCV.

    Args:
        original_image (numpy.ndarray): The original image.
        processed_images (dict): A dictionary of processed images.
    """
    cv2.imshow('Original Image', original_image)

    for technique, image in processed_images.items():
        cv2.imshow(technique, image)

    # Wait for a key press
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

def save_images(processed_images):
    """
    Save the processed images to file.

    Args:
        processed_images (dict): A dictionary of processed images.
    """
    for technique, image in processed_images.items():
        cv2.imwrite(f'{technique}.jpg', image)

def main():
    try:
        # Load the image
        image_path = input("Enter the image file path: ")
        image = load_image(image_path)

        # Get user input for image processing techniques
        techniques = input("Enter the image processing techniques (separated by commas): ").split(',')

        # Apply image processing techniques in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            processed_images = executor.submit(apply_image_processing, image, techniques)
            processed_images = processed_images.result()

        # Display the original and processed images
        display_images(image, processed_images)

        # Save the processed images
        save_images(processed_images)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
