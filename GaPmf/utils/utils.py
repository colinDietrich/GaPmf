import numpy as np
import scipy.constants as const
from typing import Callable

def waveBW_to_spectrBW(FWHM: float, lambda_s_0: float) -> float:
    """
    Converts a wavelength bandwidth (FWHM) to a spectral bandwidth.

    Args:
        FWHM (float): Full Width at Half Maximum (FWHM) of the wavelength in meters.
        lambda_s_0 (float): Central wavelength in meters.

    Returns:
        float: Spectral bandwidth in Hz.
    """
    convert = lambda x: 2 * np.pi * const.c / x
    frequency1 = convert(lambda_s_0 - FWHM / 2)
    frequency2 = convert(lambda_s_0 + FWHM / 2)
    return np.abs(frequency1 - frequency2)

def gaussian_function(wavelength_array: np.ndarray, wavelength_central: float, FWHM: float) -> np.ndarray:
    """
    Generates a Gaussian function centered at a specified wavelength.

    Args:
        wavelength_array (np.ndarray): Array of wavelengths.
        wavelength_central (float): Central wavelength in meters.
        FWHM (float): Full Width at Half Maximum of the Gaussian in meters.

    Returns:
        np.ndarray: Normalized Gaussian function.
    """
    std = FWHM / (2 * np.sqrt(2 * np.log(2)))
    gaussian = np.exp(-(wavelength_array - wavelength_central) ** 2 / (2 * std ** 2))
    gaussian_normalized = gaussian / np.linalg.norm(np.abs(gaussian))
    return gaussian_normalized

def rectangle_function(wavelength_array: np.ndarray, center_wavelength: float, bandwidth: float) -> np.ndarray:
    """
    Generates a rectangular function centered at a specified wavelength.

    Args:
        wavelength_array (np.ndarray): Array of wavelengths.
        center_wavelength (float): Central wavelength in meters.
        bandwidth (float): Bandwidth of the rectangular function in meters.

    Returns:
        np.ndarray: Normalized rectangular function.
    """
    half_bandwidth = bandwidth / 2
    lower_bound = center_wavelength - half_bandwidth
    upper_bound = center_wavelength + half_bandwidth

    # Create the rectangular function
    rect_function = np.where((wavelength_array >= lower_bound) & (wavelength_array <= upper_bound), 1.0, 0.0)
    rect_function_normalized = rect_function / np.linalg.norm(np.abs(rect_function))

    return rect_function_normalized

def pump_gaussian_function(signal_frequency_array: np.ndarray, idler_frequency_array: np.ndarray, 
                           pump_frequency_central: float, FWHM: float) -> np.ndarray:
    """
    Generates a Gaussian function for the pump in the frequency domain.

    Args:
        signal_frequency_array (np.ndarray): Array of signal frequencies in Hz.
        idler_frequency_array (np.ndarray): Array of idler frequencies in Hz.
        pump_frequency_central (float): Central frequency of the pump in Hz.
        FWHM (float): Full Width at Half Maximum of the Gaussian in Hz.

    Returns:
        np.ndarray: Normalized Gaussian function in the frequency domain.
    """
    std = FWHM / (2 * np.sqrt(2 * np.log(2)))
    gaussian = np.exp(-(signal_frequency_array + idler_frequency_array - pump_frequency_central) ** 2 / (2 * std ** 2))
    gaussian_normalized = gaussian / np.linalg.norm(np.abs(gaussian))
    return gaussian_normalized

def pump_array_to_matrix(lambda_func: Callable, list1: np.ndarray, list2: np.ndarray, 
                         lambda_p_0: float, FWHM: float) -> np.ndarray:
    """
    Converts a pump function defined in the wavelength domain to a frequency-dependent matrix.

    Args:
        lambda_func (Callable): Function defining the pump in the wavelength domain.
        list1 (np.ndarray): Array of wavelengths for the first axis (signal or idler).
        list2 (np.ndarray): Array of wavelengths for the second axis (signal or idler).
        lambda_p_0 (float): Central pump wavelength in meters.
        FWHM (float): Full Width at Half Maximum of the pump in meters.

    Returns:
        np.ndarray: Pump function in the frequency domain as a 2D matrix.
    """
    convert = lambda x: 2 * np.pi * const.c / x
    FWHM_freq = waveBW_to_spectrBW(FWHM, lambda_p_0)
    A = []
    for idx1 in list1:
        A_row = []
        for idx2 in list2:
            A_row.append(lambda_func(convert(idx1), convert(idx2), convert(lambda_p_0), FWHM_freq))
        A.append(A_row)
    return np.array(A)

def phase_mismatch_array_to_matrix(Delta_k: Callable[[float, float], float], 
                                   omega_s: np.ndarray, omega_i: np.ndarray) -> np.ndarray:
    """
    Converts a phase mismatch function to a frequency-dependent matrix.

    Args:
        Delta_k (Callable[[float, float], float]): Phase mismatch function.
        omega_s (np.ndarray): Array of signal angular frequencies in rad/s.
        omega_i (np.ndarray): Array of idler angular frequencies in rad/s.

    Returns:
        np.ndarray: Phase mismatch function as a 2D matrix.
    """
    pm_array = []
    for ws in omega_s:
        pm_row = []
        for wi in omega_i:
            pm_row.append(Delta_k(ws, wi))
        pm_array.append(pm_row)
    return np.array(pm_array)

def gaussian_target_pmf(signal_FWHM: float, idler_FWHM: float, 
                        signal_wavelength_array: np.ndarray, idler_wavelength_array: np.ndarray,
                        signal_central_wavelength: float, idler_central_wavelength: float, 
                        pump_central_wavelength: float, phase_mismatch_function: Callable) -> tuple:
    """
    Generates target phase matching functions (PMF) with Gaussian profiles for signal and idler.

    Args:
        signal_FWHM (float): FWHM of the signal in meters.
        idler_FWHM (float): FWHM of the idler in meters.
        signal_wavelength_array (np.ndarray): Array of signal wavelengths in meters.
        idler_wavelength_array (np.ndarray): Array of idler wavelengths in meters.
        signal_central_wavelength (float): Central wavelength of the signal in meters.
        idler_central_wavelength (float): Central wavelength of the idler in meters.
        pump_central_wavelength (float): Central wavelength of the pump in meters.
        phase_mismatch_function (Callable): Function to calculate the phase mismatch.

    Returns:
        tuple: Target signal function, target idler function, signal phase mismatch array, idler phase mismatch array.
    """
    signal_phase_mismatch_array = []
    for l in signal_wavelength_array:
        w = 2 * np.pi * const.c / l
        wp = 2 * np.pi * const.c / pump_central_wavelength
        signal_phase_mismatch_array.append(phase_mismatch_function(w, wp - w))
    signal_phase_mismatch_array = np.array(signal_phase_mismatch_array)

    idler_phase_mismatch_array = []
    for l in idler_wavelength_array:
        w = 2 * np.pi * const.c / l
        wp = 2 * np.pi * const.c / pump_central_wavelength
        idler_phase_mismatch_array.append(phase_mismatch_function(wp - w, w))
    idler_phase_mismatch_array = np.array(idler_phase_mismatch_array)

    target_signal_function = gaussian_function(signal_wavelength_array, signal_central_wavelength, signal_FWHM)
    target_idler_function = gaussian_function(idler_wavelength_array, idler_central_wavelength, idler_FWHM)

    return target_signal_function, target_idler_function, signal_phase_mismatch_array, idler_phase_mismatch_array

def rectangle_target_pmf(signal_FWHM: float, idler_FWHM: float, 
                         signal_wavelength_array: np.ndarray, idler_wavelength_array: np.ndarray,
                         signal_central_wavelength: float, idler_central_wavelength: float, 
                         pump_central_wavelength: float, phase_mismatch_function: Callable) -> tuple:
    """
    Generates target phase matching functions (PMF) with rectangular profiles for signal and idler.

    Args:
        signal_FWHM (float): FWHM of the signal in meters.
        idler_FWHM (float): FWHM of the idler in meters.
        signal_wavelength_array (np.ndarray): Array of signal wavelengths in meters.
        idler_wavelength_array (np.ndarray): Array of idler wavelengths in meters.
        signal_central_wavelength (float): Central wavelength of the signal in meters.
        idler_central_wavelength (float): Central wavelength of the idler in meters.
        pump_central_wavelength (float): Central wavelength of the pump in meters.
        phase_mismatch_function (Callable): Function to calculate the phase mismatch.

    Returns:
        tuple: Target signal function, target idler function, signal phase mismatch array, idler phase mismatch array.
    """
    signal_phase_mismatch_array = []
    for l in signal_wavelength_array:
        w = 2 * np.pi * const.c / l
        wp = 2 * np.pi * const.c / pump_central_wavelength
        signal_phase_mismatch_array.append(phase_mismatch_function(w, wp - w))
    signal_phase_mismatch_array = np.array(signal_phase_mismatch_array)

    idler_phase_mismatch_array = []
    for l in idler_wavelength_array:
        w = 2 * np.pi * const.c / l
        wp = 2 * np.pi * const.c / pump_central_wavelength
        idler_phase_mismatch_array.append(phase_mismatch_function(wp - w, w))
    idler_phase_mismatch_array = np.array(idler_phase_mismatch_array)

    target_signal_function = rectangle_function(signal_wavelength_array, signal_central_wavelength, signal_FWHM)
    target_idler_function = rectangle_function(idler_wavelength_array, idler_central_wavelength, idler_FWHM)

    return target_signal_function, target_idler_function, signal_phase_mismatch_array, idler_phase_mismatch_array
