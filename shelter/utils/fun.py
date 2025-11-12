import numpy as np
import traceback

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

def sRGB_to_HSV(rgb):
    """
    Convert a set of sRGB colour values to HSV.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        rgb (array-like): sRGB value

    Returns:
        HSV (array-like): HSV value
    """
    return 1
    #X_max = np.max()

def sRGB_to_Hex(rgb):
    """
    Convert a set of sRGB colour values to a Hex code.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        rgb (array-like): sRGB value

    Returns:
        Hex (string or list of strings): Hex code
    """
    rgb = np.asarray(rgb)
    if len(rgb.shape) >= 2: 
        hexes = []
        for n in range(rgb.shape[1]):
            hex = '#' + ''.join('%02X' % round(i*255) for i in rgb[:, n])
            hexes.append(hex)
        return hexes
    return '#' + ''.join('%02X' % round(i*255) for i in rgb)

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
    return np.matmul(transfer_matrix, np.array(XYZ))

def XYZ_to_HSV(XYZ):
    """
    Convert a set of CIE XYZ colour values to HSV.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        XYZ (array-like): CIE XYZ value 

    Returns:
        HSV (array-like): HSV value
    """



def XYZ_to_Oklab(XYZ):
    """
    Convert a set of CIE XYZ colour values to Oklab.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        XYZ (array-like): CIE XYZ value

    Returns:
        Lab (array-like): Oklab value
    """

    transfer_matrix_1 = np.array([[+0.8189330101, +0.3618667424, -0.1288597137],
                                  [+0.0329845436, +0.9293118715, +0.0361456387],
                                  [+0.0482003018, +0.2643662691, +0.6338517070]])
    
    transfer_matrix_2 = np.array([[+0.2104542553, +0.7936177850, -0.0040720468],
                                  [+1.9779984951, -2.4285922050, +0.4505937099],
                                  [+0.0259040371, +0.7827717662, -0.8086757660]])
    
    return np.matmul(transfer_matrix_2, (np.matmul(transfer_matrix_1, np.array(XYZ))**(1/3)))

def sRGB_to_XYZ(rgb):
    """
    Convert a set of sRGB colour values to CIE XYZ.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        rgb (array-like): sRGB value

    Returns:
        XYZ (array-like): CIE XYZ value
    """
        
    return RGB_to_XYZ(sRGB_to_RGB(rgb))

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
    return np.matmul(transfer_matrix, np.array(RGB))

def HSV_to_XYZ(HSV):
    """
    Convert a set of HSV colour values to CIE XYZ.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        HSV (array-like): HSV value

    Returns:
        XYZ (array-like): CIE XYZ value
    """
        
    return RGB_to_XYZ(sRGB_to_RGB(HSV_to_sRGB(HSV)))

def Oklab_to_XYZ(Lab):
    """
    Convert a set of Oklab colour values to CIE XYZ.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        Lab (array-like): Oklab value

    Returns:
        XYZ (array-like): CIE XYZ value
    """

    transfer_matrix_1 = np.linalg.inv(np.array([[+0.8189330101, +0.3618667424, -0.1288597137],
                                                [+0.0329845436, +0.9293118715, +0.0361456387],
                                                [+0.0482003018, +0.2643662691, +0.6338517070]]))
    
    transfer_matrix_2 = np.linalg.inv(np.array([[+0.2104542553, +0.7936177850, -0.0040720468],
                                                [+1.9779984951, -2.4285922050, +0.4505937099],
                                                [+0.0259040371, +0.7827717662, -0.8086757660]]))
    
    return np.matmul(transfer_matrix_1, (np.matmul(transfer_matrix_2, np.array(Lab))**3))

# Append any new spaces implemented here please!
_to_XYZ_methods = {
    "sRGB": lambda v: sRGB_to_XYZ(v),
    "RGB": lambda v: RGB_to_XYZ(v),
    "HSV": lambda v: HSV_to_XYZ(v),
    "XYZ": lambda v: np.array(v),
    "Oklab": lambda v: Oklab_to_XYZ(v), 
}

_from_XYZ_methods = {
    "sRGB": lambda v: RGB_to_sRGB(XYZ_to_RGB(v)),
    "RGB": lambda v: XYZ_to_RGB(v),
    "HSV": lambda v: HSV_to_XYZ(v),
    "Hex": lambda v: sRGB_to_Hex(RGB_to_sRGB(XYZ_to_RGB(v))),
    "XYZ": lambda v: np.array(v),
    "Oklab": lambda v: XYZ_to_Oklab(v), 
}

class Colour:
    """A class for storing and manipulating colours."""

    def __init__(self, value, space, alpha=None):
        """
        Parameters:
            value (array-like of floats): 3 floats (e.g., (r, g, b) or (h, s, v))
            space (string): One of "sRGB", "RGB", "HSV", etc.
            alpha (float): Value for the alpha (opacity) channel. Defaults to 1.
        """
        value = np.asarray(value)

        if space not in _to_XYZ_methods:
            raise KeyError(f'Unsupported or invalid colour space: {space}')
            
        if alpha == None:
            if len(np.array(value).shape) == 1:
                alpha = 1
            else:
                alpha = np.ones(np.array(value).shape[1])

        self._value = _to_XYZ_methods[space](value)     # Colour value stored as an array
        self._space = "XYZ"     # All colours are internally stored in CIE XYZ space. Why is this a variable?
        self.alpha = alpha      # Alpha channel is stored separately
    
    def set_colour(self, value, space):
        """
        Set the colour in the object to a new colour.

        value (array-like): 3 floats (e.g., (r, g, b) or (h, s, v))
        space (string): One of "sRGB", "RGB", "HSV", etc.
        """
        value = np.asarray(value)

        if space not in _to_XYZ_methods:
            raise KeyError(f'Unsupported or invalid colour space: {space}')
            
        if alpha == None:
            alpha = np.ones(len(np.asarray([value[0]])))
        
        self._value = _to_XYZ_methods[space](value)     # Colour value stored as an array

    def set_alpha(self, alpha):
        """Set alpha value of colour. Here for consistency with set_colour."""
        self.alpha = alpha

    def get_colour(self, space, return_alpha=False):
        """
        Get the colour value in the desired colour space.

        space (string): Colour space to output into.
        return_alpha (bool, default False): If True, returns (colour, alpha).
        """
        if space not in _from_XYZ_methods:
            raise KeyError(f'Unsupported or invalid colour space: {space}')

        out = _from_XYZ_methods[space](self._value)
        if return_alpha:
            return out, self.alpha
        return out
    
    def get_alpha(self):
        """Get alpha value of colour. Here for consistency with get_colour."""
        return self.alpha


class Gradient:
    """A class for creating gradients through a desired colour space."""

    def __init__(self, colours, positions=None, interp_space='Oklab'):
        """
        Parameters:
            colours (array-like of Colours): Set of Colour objects to create the gradient with.
            positions (array-like of floats): Positions of the colour stops. Values must be defined between 0 and 1.
            By default colours will be spaced out equally.
            interp_space (string): Colour space to interpolate the gradient within.
        """
        n = len(colours)
        if n <= 1:
            raise ValueError("Gradient requires at least 2 colours")    # Could maybe use 1 but that would be pointless
        
        if positions is None:
            positions = np.linspace(0, 1, n)
        else:
            positions = np.asarray(positions, dtype=float)
            if positions.shape[0] != n:
                raise ValueError(f"Number of positions ({positions.shape[0]}) does not match number of colours ({n})")
            if np.any(positions < 0) or np.any(positions > 1):
                raise ValueError("Positions must be between 0 and 1")
            
        self.interp_space = interp_space
        if self.interp_space not in _to_XYZ_methods:
            raise KeyError(f"Unsupported or invalid interpolation space: {interp_space}")
        elif self.interp_space not in _from_XYZ_methods:
            raise KeyError(f"Unsupported or invalid interpolation space: {interp_space}")
        
        # Sort the colours by position
        sort_order = np.argsort(positions)
        self.positions = np.asarray(positions)[sort_order]
        self.colours = np.asarray(colours)[sort_order]

    def _sample_gradient(self, t):
        """
        Sample the gradient at a point (or set of points).
        If the gradient is empty, picked colours will be solid black.
        """
        

    def add_stop(self, colour=None, position=None):
        """
        Add a new colour stop to the gradient.
        If colour is None then new colour will be interpolated from the existing gradient at the chosen position.
        If position is None then the stop will be added in the centre of the largest gap.

        Parameters:
            colour (Colour): Colour object to assign to the stop.
            position (float): Position of the colour stop. Value must be defined between 0 and 1.
        """
        if position == None:
            
        

colour = Colour([0.1, 0.1, 0.1], "RGB")
print(colour.get_colour("Oklab"))
print(colour.get_colour("Hex"))
print(colour.get_alpha())
        
colour = Colour([np.linspace(0.1, 0.1, 8), np.linspace(0.1, 0.1, 8), np.linspace(0.1, 0.1, 8)], "RGB")
print(colour.get_colour("Oklab"))
print(colour.get_colour("Hex"))
print(colour.get_alpha())

colour1 = Colour([0, 0.1, 0.2], "RGB")
colour2 = Colour([1, 0.1, 0.2], "RGB")
gradient = Gradient([colour1, colour2])