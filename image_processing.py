import numpy as np
try:
    import cv2
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2


def preprocess_image(image, clip_limit=2.0):
    # Convert to YCrCb color space for better contrast enhancement
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_channel_stretched = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX)

    # # Apply CLAHE (Adaptive Histogram Equalization) on the Y (luminance) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(3, 3))
    y_clahe = clahe.apply(y_channel_stretched)

    # # Merge and convert back to RGB
    image = cv2.merge([y_clahe, cr, cb])
    image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2RGB)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Aplly sharpening
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

    return sharpened


def resize_image(image, dsize=(200, 200)):
    resized_image = cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_LANCZOS4)
    return resized_image
