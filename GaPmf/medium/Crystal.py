import numpy as np
from scipy.constants import c
from random import choices, randint
import math
from scipy.ndimage import zoom
from typing import Tuple, Callable
from GaPmf.utils.utils import phase_mismatch_array_to_matrix, pump_array_to_matrix

class Crystal:
    def __init__(self,
                 domain_width: float,
                 initial_configuration: str,
                 maximum_length: float,
                 minimum_length: float,
                 mode: int,
                 phase_mismatch_array: np.ndarray = None,
                 pmf_target: np.ndarray = None,
                 signal_phase_mismatch_array: np.ndarray = None,
                 idler_phase_mismatch_array: np.ndarray = None,
                 signal_pmf_target: np.ndarray = None,
                 idler_pmf_target: np.ndarray = None,
                 domain_values_custom: np.ndarray = None) -> None:
        """
        Initializes the Crystal class with given parameters.
        
        Args:
            domain_width (float): Width of each domain in the crystal [m].
            initial_configuration (str): Initial configuration of the crystal's design (e.g., 'ppln', 'random', 'custom').
            maximum_length (float): Maximum length of the crystal [m].
            minimum_length (float): Minimum length of the crystal [m].
            mode (int): Mode of operation (1 or 2).
            phase_mismatch_array (np.ndarray, optional): Phase mismatch array for mode 1.
            pmf_target (np.ndarray, optional): Target PMF for mode 1.
            signal_phase_mismatch_array (np.ndarray, optional): Signal phase mismatch array for mode 2.
            idler_phase_mismatch_array (np.ndarray, optional): Idler phase mismatch array for mode 2.
            signal_pmf_target (np.ndarray, optional): Target signal PMF for mode 2.
            idler_pmf_target (np.ndarray, optional): Target idler PMF for mode 2.
            domain_values_custom (np.ndarray, optional): Custom domain values for the crystal configuration.
        """
        self.initial_configuration = initial_configuration
        self.domain_width = domain_width
        self.maximum_length = maximum_length
        self.minimum_length = minimum_length
        self.mode = mode

        # Validate mode and required arrays
        if self.mode == 1:
            if phase_mismatch_array is not None and pmf_target is not None:
                self.phase_mismatch_array = phase_mismatch_array
                self.pmf_target = pmf_target
            else:
                raise ValueError('Phase mismatch array and PMF target must be provided for mode 1.')
        elif self.mode == 2:
            if (signal_phase_mismatch_array is not None and idler_phase_mismatch_array is not None and
                signal_pmf_target is not None and idler_pmf_target is not None):
                self.signal_phase_mismatch_array = signal_phase_mismatch_array
                self.idler_phase_mismatch_array = idler_phase_mismatch_array
                self.signal_pmf_target = signal_pmf_target
                self.idler_pmf_target = idler_pmf_target
            else:
                raise ValueError('Signal phase mismatch array, idler phase mismatch array, signal PMF target, and idler PMF target must be provided for mode 2.')
        else:
            raise ValueError('Mode must be 1 or 2.')

        self.domain_values = None
        self.domain_values_custom = domain_values_custom
        self.domain_bounds_custom = None
        self.number_of_domains = None
        self.length = None
        self.z_grid = None
        self.level = 1

        # Initialize the crystal based on the initial configuration
        self.initialization()

    def initialization(self) -> None:
        """
        Initializes the crystal's domain values and length based on the initial configuration.
        """
        if self.initial_configuration == "ppln":
            # Generate a random number of domains between given limits for 'ppln' configuration
            if self.minimum_length != self.maximum_length:
                self.number_of_domains = randint(math.ceil(self.minimum_length / self.domain_width),
                                                 math.floor(self.maximum_length / self.domain_width))
            else:
                self.number_of_domains = math.floor(self.maximum_length / self.domain_width)
            self.domain_values = [1, -1] * (self.number_of_domains // 2)
        elif self.initial_configuration == "custom":
            # Use custom domain values if provided
            self.number_of_domains = len(self.domain_values_custom)
            self.domain_values = self.domain_values_custom
        else:
            # Generate a random number of domains for 'random' configuration
            if self.minimum_length != self.maximum_length:
                self.number_of_domains = randint(math.ceil(self.minimum_length / self.domain_width),
                                                 math.floor(self.maximum_length / self.domain_width))
            else:
                self.number_of_domains = math.floor(self.maximum_length / self.domain_width)
            self.domain_values = choices([1, -1], k=self.number_of_domains)

        # Calculate the crystal length and grid positions
        self.length = self.number_of_domains * self.domain_width
        self.z_grid = np.arange(-self.length / 2, -self.length / 2 + (self.number_of_domains + 1) * self.domain_width, self.domain_width)
        
        # Calculate the PMF
        self.update()

    def update(self, level: int = 1) -> None:
        """
        Updates the crystal's properties, including domain values, length, and PMF.
        
        Args:
            level (int): The refinement level for the domain width. Default is 1.
        """
        if level != self.level:
            self.level = level
            self.domain_width /= 2
            self.domain_values = self.double_length_array(self.domain_values)
        
        self.number_of_domains = len(self.domain_values)
        self.length = self.number_of_domains * self.domain_width
        self.z_grid = np.arange(-self.length / 2, -self.length / 2 + (self.number_of_domains + 1) * self.domain_width, self.domain_width)

        if self.mode == 1:
            self.pmf = self.phase_matching_function()
            self.pmf /= np.linalg.norm(np.abs(self.pmf))
            self.mse_abs = self.mean_squared_error(np.abs(self.pmf_target), np.abs(self.pmf))
        else:
            self.pmf_signal, self.pmf_idler = self.phase_mismatch_with_frequency_array()
            self.pmf_signal /= np.linalg.norm(np.abs(self.pmf_signal))
            self.pmf_idler /= np.linalg.norm(np.abs(self.pmf_idler))
            self.mse_abs_signal = self.mean_squared_error(np.abs(self.signal_pmf_target), np.abs(self.pmf_signal))
            self.mse_abs_idler = self.mean_squared_error(np.abs(self.idler_pmf_target), np.abs(self.pmf_idler))

    def mean_squared_error(self, array1: np.ndarray, array2: np.ndarray) -> float:
        """
        Calculates the mean squared error between two arrays.
        
        Args:
            array1 (np.ndarray): First array.
            array2 (np.ndarray): Second array.
        
        Returns:
            float: The mean squared error between the arrays.
        """
        array1 /= np.linalg.norm(np.abs(array1))
        array2 /= np.linalg.norm(np.abs(array2))
        return np.square(np.subtract(array1, array2)).mean()

    def double_length_array(self, array: list) -> list:
        """
        Doubles the length of the given array by repeating each element.
        
        Args:
            array (list): The array to double in length.
        
        Returns:
            list: The doubled-length array.
        """
        return [element for element in array for _ in range(2)]

    def phase_matching_with_matrix(self, phase_mismatch_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the phase matching function (PMF) using a phase mismatch matrix.
        
        Args:
            phase_mismatch_matrix (np.ndarray): The phase mismatch matrix.
        
        Returns:
            np.ndarray: The calculated PMF.
        """
        pmf_one_domain = lambda z1, z2: (1 / self.length) * 1j * \
            (np.exp(1j * phase_mismatch_matrix * z1) - np.exp(1j * phase_mismatch_matrix * z2)) / phase_mismatch_matrix

        pmf = 0
        for idz in range(len(self.domain_values) - 1):
            pmf += self.domain_values[idz] * pmf_one_domain(self.z_grid[idz], self.z_grid[idz + 1])

        return pmf

    def phase_mismatch_with_frequency_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the phase matching function (PMF) for both signal and idler frequencies.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: The PMF for signal and idler frequencies.
        """
        pmf_one_domain_signal = lambda z1, z2: (1 / self.length) * 1j * \
            (np.exp(1j * self.signal_phase_mismatch_array * z1) - np.exp(1j * self.signal_phase_mismatch_array * z2)) / self.signal_phase_mismatch_array

        pmf_one_domain_idler = lambda z1, z2: (1 / self.length) * 1j * \
            (np.exp(1j * self.idler_phase_mismatch_array * z1) - np.exp(1j * self.idler_phase_mismatch_array * z2)) / self.idler_phase_mismatch_array

        pmf_signal = 0
        pmf_idler = 0
        for idz in range(len(self.domain_values) - 1):
            pmf_signal += self.domain_values[idz] * pmf_one_domain_signal(self.z_grid[idz], self.z_grid[idz + 1])
            pmf_idler += self.domain_values[idz] * pmf_one_domain_idler(self.z_grid[idz], self.z_grid[idz + 1])

        return pmf_signal, pmf_idler

    def phase_matching_function(self) -> np.ndarray:
        """
        Calculates the phase matching function (PMF) for the crystal.
        
        Returns:
            np.ndarray: The calculated PMF.
        """
        pmf_one_domain = lambda z1, z2: (1 / self.length) * 1j * \
            (np.exp(1j * self.phase_mismatch_array * z1) - np.exp(1j * self.phase_mismatch_array * z2)) / self.phase_mismatch_array

        pmf = 0
        for idz in range(len(self.domain_values) - 1):
            pmf += self.domain_values[idz] * pmf_one_domain(self.z_grid[idz], self.z_grid[idz + 1])

        return pmf

    def compute_jsa(self, signal_wavelength_array: np.ndarray, idler_wavelength_array: np.ndarray,
                    pump_central_wavelength: float, FWHM: float,
                    pump_function: Callable[[np.ndarray, float, float], np.ndarray],
                    phase_mismatch_function: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        """
        Computes the Joint Spectral Amplitude (JSA) for the crystal.
        
        Args:
            signal_wavelength_array (np.ndarray): Array of signal wavelengths.
            idler_wavelength_array (np.ndarray): Array of idler wavelengths.
            pump_central_wavelength (float): Central wavelength of the pump.
            FWHM (float): Full Width at Half Maximum of the pump.
            pump_function (Callable[[np.ndarray, float, float], np.ndarray]): Function to calculate the pump spectrum.
            phase_mismatch_function (Callable[[np.ndarray, np.ndarray], np.ndarray]): Function to calculate the phase mismatch.
        """
        phase_mismatch_matrix = np.array(phase_mismatch_array_to_matrix(phase_mismatch_function,
                                                                        2 * np.pi * c / signal_wavelength_array,
                                                                        2 * np.pi * c / idler_wavelength_array))
        self.spectral_pmf_matrix = self.phase_matching_with_matrix(phase_mismatch_matrix)
        self.spectral_pump_matrix = pump_array_to_matrix(pump_function, signal_wavelength_array,
                                                         idler_wavelength_array, pump_central_wavelength, FWHM)
        self.jsa = self.spectral_pump_matrix * self.spectral_pmf_matrix

    def compute_purity(self) -> float:
        """
        Computes the purity of the generated photon pairs based on the Joint Spectral Amplitude (JSA).
        
        Returns:
            float: The calculated cooperativity (K) indicating the purity.
        """
        jsa_reshaped = zoom(self.jsa, (100 / self.jsa.shape[0], 100 / self.jsa.shape[1]), order=3)
        u, s, vh = np.linalg.svd(jsa_reshaped, full_matrices=True)

        s /= np.linalg.norm(s)
        self.coop = 1 / sum(val**4 for val in s)

        print(f"Cooperativity K = {self.coop}")
        return self.coop
