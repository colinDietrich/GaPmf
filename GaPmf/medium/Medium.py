import numpy as np
from scipy.constants import c
from typing import List, Callable
from GaPmf.utils.data import LiNbO3, KTP

class Medium:
    def __init__(self,
                 medium: str,
                 signal: str,
                 idler: str,
                 pump: str,
                 pump_wavelength: float) -> None:
        """
        Initializes the Medium class with given parameters.
        
        Args:
            medium (str): The type of crystal medium (e.g., 'LiNbO3', 'KTP').
            signal (str): The signal orientation or axis.
            idler (str): The idler orientation or axis.
            pump (str): The pump orientation or axis.
            pump_wavelength (float): The wavelength of the pump in meters.
        """
        self.pump_wavelength = pump_wavelength
        self.pump_angular_frequency = 2 * np.pi * c / pump_wavelength

        # Initialize the crystal medium properties
        if medium == 'LiNbO3':
            self.data_medium = LiNbO3
            self.orientations = ['x', 'y']
        elif medium == 'KTP':
            self.data_medium = KTP
            self.orientations = ['x', 'y', 'z']
        else:
            raise ValueError('Unknown medium for the crystal.')

        # Calculate refractive indices for signal, idler, and pump
        self.n_s = self.sellmeier(self.data_medium[signal])
        self.n_i = self.sellmeier(self.data_medium[idler])
        self.n_p = self.sellmeier(self.data_medium[pump])

        # Calculate the wave vectors for signal, idler, and pump
        self.k_s = lambda freq: freq / c * self.n_s(freq)
        self.k_i = lambda freq: freq / c * self.n_i(freq)
        self.k_p = lambda freq: freq / c * self.n_p(freq)

        # Compute the phase mismatch function
        self.phase_mismatch_function = self.wavevector_mismatch()

        # Calculate the domain width based on the phase mismatch
        dk = self.phase_mismatch_function(self.pump_angular_frequency / 2, self.pump_angular_frequency / 2)
        self.domain_width = np.abs(np.pi / dk)

    def wavevector_mismatch(self) -> Callable[[float, float], float]:
        """
        Returns a function that calculates the wavevector mismatch for given signal and idler frequencies.
        
        Returns:
            Callable[[float, float], float]: A function that calculates the wavevector mismatch.
        """
        return lambda ws, wi: self.k_s(ws) + self.k_i(wi) - self.k_p(ws + wi)

    def sellmeier(self, A: List[float]) -> Callable[[float], float]:
        """
        Returns the refractive index as a function of frequency using the Sellmeier equation.
        
        Args:
            A (List[float]): List of Sellmeier coefficients.
        
        Returns:
            Callable[[float], float]: Function that calculates the refractive index for a given frequency.
        """
        return lambda x: np.sqrt(A[0] + A[1] / ((2 * np.pi * c / x * 1e6) ** 2 - A[2]) - A[3] * (2 * np.pi * c / x * 1e6) ** 2)