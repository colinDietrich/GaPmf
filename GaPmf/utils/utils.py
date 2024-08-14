import numpy as np
import scipy.constants as const
import numdifftools as nd

def waveBW_to_spectrBW(FWHM, lambda_s_0):
    convert = lambda x: 2*np.pi*const.c/x
    frequency1 = convert(lambda_s_0-FWHM/2)
    frequency2 = convert(lambda_s_0+FWHM/2)
    return np.abs(frequency1-frequency2)

# define gaussian function
def gaussian_function(wavelength_array, wavelength_central, FWHM):
    std = FWHM / (2 * np.sqrt(2 * np.log(2)))
    gaussian = np.exp(-(wavelength_array-wavelength_central)**2/(2*std**2))
    gaussian_normalized = gaussian / np.linalg.norm(np.abs(gaussian))
    return gaussian_normalized

def rectangle_function(wavelength_array, center_wavelength, bandwidth):
    half_bandwidth = bandwidth / 2
    lower_bound = center_wavelength - half_bandwidth
    upper_bound = center_wavelength + half_bandwidth

    # Create the rectangular function
    rect_function = np.where((wavelength_array >= lower_bound) & (wavelength_array <= upper_bound), 1.0, 0.0)
    rect_function_normalized = rect_function / np.linalg.norm(np.abs(rect_function))

    return rect_function_normalized

def pump_gaussian_function(signal_frequency_array, idler_frequency_array, pump_frequency_central, FWHM):
    std = FWHM / (2 * np.sqrt(2 * np.log(2)))
    gaussian = np.exp(-(signal_frequency_array + idler_frequency_array - pump_frequency_central)**2/(2*std**2))
    gaussian_normalized = gaussian / np.linalg.norm(np.abs(gaussian))
    return gaussian

# convert pump array to frequency (ws, wi) dependent matrix
def pump_array_to_matrix(lambda_func,list1,list2,lambda_p_0,FWHM):
    convert = lambda x: 2*np.pi*const.c/x
    FWHM_freq = waveBW_to_spectrBW(FWHM, lambda_p_0)
    A =[]
    for idx1 in list1:
        A_row = []
        for idx2 in list2:
            A_row += [lambda_func(convert(idx1),convert(idx2),convert(lambda_p_0),FWHM_freq)]
        A += [A_row]
    return A

# convert phase mismatch array to frequency (ws, wi) dependent matrix
def phase_mismatch_array_to_matrix(Delta_k, omega_s, omega_i):
  pm_array =[]
  for ws in omega_s:
      pm_row = []
      for wi in omega_i:
          pm_row += [Delta_k(ws, wi)]
      pm_array += [pm_row]
  return pm_array

def gaussian_target_pmf(signal_FWHM, idler_FWHM,
                        signal_wavelength_array,
                        idler_wavelength_array,
                        signal_central_wavelength,
                        idler_central_wavelength,
                        pump_central_wavelength,
                        phase_mismatch_function):
    signal_phase_mismatch_array = []
    for l in signal_wavelength_array:
        w = 2 * np.pi * const.c / l
        wp = 2 * np.pi * const.c / pump_central_wavelength
        signal_phase_mismatch_array.append(phase_mismatch_function(w, wp-w))
    signal_phase_mismatch_array = np.array(signal_phase_mismatch_array)

    idler_phase_mismatch_array = []
    for l in idler_wavelength_array:
        w = 2 * np.pi * const.c / l
        wp = 2 * np.pi * const.c / pump_central_wavelength
        idler_phase_mismatch_array.append(phase_mismatch_function(wp-w, w))
    idler_phase_mismatch_array = np.array(idler_phase_mismatch_array)

    target_signal_function = gaussian_function(signal_wavelength_array, signal_central_wavelength, signal_FWHM)
    target_idler_function = gaussian_function(idler_wavelength_array, idler_central_wavelength, idler_FWHM)

    return target_signal_function, target_idler_function, signal_phase_mismatch_array, idler_phase_mismatch_array

def rectangle_target_pmf(signal_FWHM, idler_FWHM,
                        signal_wavelength_array,
                        idler_wavelength_array,
                        signal_central_wavelength,
                        idler_central_wavelength,
                        pump_central_wavelength,
                        phase_mismatch_function):
    signal_phase_mismatch_array = []
    for l in signal_wavelength_array:
        w = 2 * np.pi * const.c / l
        wp = 2 * np.pi * const.c / pump_central_wavelength
        signal_phase_mismatch_array.append(phase_mismatch_function(w, wp-w))
    signal_phase_mismatch_array = np.array(signal_phase_mismatch_array)

    idler_phase_mismatch_array = []
    for l in idler_wavelength_array:
        w = 2 * np.pi * const.c / l
        wp = 2 * np.pi * const.c / pump_central_wavelength
        idler_phase_mismatch_array.append(phase_mismatch_function(wp-w, w))
    idler_phase_mismatch_array = np.array(idler_phase_mismatch_array)

    target_signal_function = rectangle_function(signal_wavelength_array, signal_central_wavelength, signal_FWHM)
    target_idler_function = rectangle_function(idler_wavelength_array, idler_central_wavelength, idler_FWHM)

    return target_signal_function, target_idler_function, signal_phase_mismatch_array, idler_phase_mismatch_array