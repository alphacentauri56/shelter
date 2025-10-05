import numpy as np

def sRGB_to_RGB(rgb):
    """
    Convert a set of sRGB colour values to Linear RGB.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        rgb (array-like): sRGB value

    Returns:
        RGB (array-like): Linear RGB value
    """

    return np.where(rgb>0.04045, ((rgb+0.055)/1.055)**2.4, rgb/12.92)

def RGB_to_sRGB(RGB):
    """
    Convert a set of Linear RGB colour values to sRGB.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        RGB (array-like): Linear RGB value

    Returns:
        rgb (array-like): sRGB value
    """

    return np.where(RGB>0.0031308, ((RGB**(1/2.4))*1.055)-0.055, RGB*12.92)

def HSV_to_sRGB(HSV):
    """
    Convert a set of HSV colour values to sRGB.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        HSV (array-like): HSV value

    Returns:
        rgb (array-like): sRGB value
    """

    def HSV_fn(HSV, n):
        k = np.mod(n + (HSV[0]*6), 6)
        return HSV[2] - (HSV[2] * HSV[1] * max(0, min(k, 4-k, 1)))
    
    return np.array([HSV_fn(HSV, 5), HSV_fn(HSV, 3), HSV_fn(HSV, 1)])

def RGB_to_XYZ(RGB):
    """
    Convert a set of Linear RGB colour values to CIE XYZ.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        RGB (array-like): Linear RGB value

    Returns:
        XYZ (array-like): CIE XYZ value
    """
        
    transfer_matrix = np.array([[0.4124, 0.3576, 0.1805],
                                [0.2126, 0.7152, 0.0722],
                                [0.0193, 0.1192, 0.9505]])
    return np.matmul(transfer_matrix, np.array(RGB).T)

def XYZ_to_RGB(XYZ):
    """
    Convert a set of CIE XYZ colour values to Linear RGB.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        XYZ (array-like): CIE XYZ value 

    Returns:
        RGB (array-like): Linear RGB value
    """
        
    transfer_matrix = np.array([[+3.2406255, -1.5372080, -0.4986286],
                                [-0.9689307, +1.8757561, +0.0415175],
                                [+0.0557101, -0.2040211, +1.0569959]])
    return np.matmul(transfer_matrix, np.array(XYZ.T))

HSV = np.array([0.1, 0.3, 0.8])
RGB = HSV_to_sRGB(HSV)
print(RGB)

XYZ = RGB_to_XYZ(RGB)
print(XYZ)


class Colour:
    """A class for storing and manipulating colours."""
    _to_XYZ_methods = {
        "RGB": lambda v: RGB_to_XYZ(v)
    }

    def __init__(self, value, space):
        """
        values: tuple/list/array of 3 floats (e.g., (r, g, b) or (h, s, v))
        space: string, one of "sRGB", "RGB", "HSV", etc.
        """
        
