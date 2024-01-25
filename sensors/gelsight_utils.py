import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def show_normalized_img(name, img):
    """
    Display a normalized image using OpenCV.

    Parameters:
    - name (str): The name of the window.
    - img (numpy.ndarray): The input image.

    Returns:
    - numpy.ndarray: The normalized image.
    """
    draw = img.copy()
    draw -= np.min(draw)
    draw = draw / np.max(draw)
    cv2.imshow(name, draw)
    return draw

def gkern2D(kernlen=21, nsig=3):
    """
    Returns a 2D Gaussian kernel array.

    Parameters:
    - kernlen (int): Size of the kernel (default is 21).
    - nsig (float): Standard deviation of the Gaussian distribution (default is 3).

    Returns:
    - numpy.ndarray: 2D Gaussian kernel.
    """
    inp = np.zeros((kernlen, kernlen))
    inp[kernlen // 2, kernlen // 2] = 1
    return gaussian_filter(inp, nsig)

def gauss_noise(image, sigma):
    """
    Add Gaussian noise to an image.

    Parameters:
    - image (numpy.ndarray): Input image.
    - sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
    - numpy.ndarray: Noisy image.
    """
    row, col = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy

def derivative(mat, direction):
    """
    Compute the derivative of an image along the specified direction.

    Parameters:
    - mat (numpy.ndarray): Input image.
    - direction (str): Derivative direction ('x' or 'y').

    Returns:
    - numpy.ndarray: Image derivative.
    """
    assert (direction == 'x' or direction == 'y'), "The derivative direction must be 'x' or 'y'"
    kernel = None
    if direction == 'x':
        kernel = [[-1.0, 0.0, 1.0]]
    elif direction == 'y':
        kernel = [[-1.0], [0.0], [1.0]]
    kernel = np.array(kernel, dtype=np.float64)
    return cv2.filter2D(mat, -1, kernel) / 2.0

def tangent(mat):
    """
    Compute the tangent vector of an image.

    Parameters:
    - mat (numpy.ndarray): Input image.

    Returns:
    - numpy.ndarray: Tangent vector of the image.
    """
    dx = derivative(mat, 'x')
    dy = derivative(mat, 'y')
    img_shape = np.shape(mat)
    _1 = np.repeat([1.0], img_shape[0] * img_shape[1]).reshape(img_shape).astype(dx.dtype)
    unnormalized = cv2.merge((-dx, -dy, _1))
    norms = np.linalg.norm(unnormalized, axis=2)
    return (unnormalized / np.repeat(norms[:, :, np.newaxis], 3, axis=2))

def solid_color_img(color, size):
    """
    Create a solid color image.

    Parameters:
    - color (tuple): RGB color tuple.
    - size (tuple): Size of the image (height, width).

    Returns:
    - numpy.ndarray: Solid color image.
    """
    image = np.zeros(size + (3,), np.float64)
    image[:] = color
    return image

def add_overlay(rgb, alpha, color):
    """
    Add an overlay to an RGB image.

    Parameters:
    - rgb (numpy.ndarray): RGB image.
    - alpha (numpy.ndarray): Alpha channel for the overlay.
    - color (tuple): RGB color tuple for the overlay.

    Returns:
    - numpy.ndarray: Image with overlay.
    """
    s = np.shape(alpha)
    opacity3 = np.repeat(alpha, 3).reshape((s[0], s[1], 3))
    overlay = solid_color_img(color, s)
    foreground = opacity3 * overlay
    background = (1.0 - opacity3) * rgb.astype(np.float64)
    res = background + foreground
    res[res > 255.0] = 255.0
    res[res < 0.0] = 0.0
    res = res.astype(np.uint8)
    return res
