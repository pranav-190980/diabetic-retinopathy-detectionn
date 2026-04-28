import cv2
import numpy as np

def preprocess_image(img_path, size=(224, 224)):
    """
    Reads image, resizes and normalizes it
    """

    img = cv2.imread(img_path)

    # Resize image
    img = cv2.resize(img, size)

    # Normalize pixel values (0-1)
    img = img / 255.0

    # Convert HWC → CHW (PyTorch format)
    img = np.transpose(img, (2, 0, 1))

    return img
