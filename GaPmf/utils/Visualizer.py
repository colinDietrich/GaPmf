import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.constants as const
import numpy as np

class Visualizer:
    def __init__(self):
        """
        Initializes the Visualizer class for plotting various scientific data.
        """
        pass

    def plot_pmf(self, delta_k_pmf: np.ndarray, delta_k_pmf0: float, pmf: np.ndarray, 
                 save: bool = False, name: str = 'polling_profile', width: float = 0.5) -> None:
        """
        Plots the phase matching function (PMF) as a function of delta k.

        Args:
            delta_k_pmf (np.ndarray): Array of delta k values.
            delta_k_pmf0 (float): Central delta k value.
            pmf (np.ndarray): Array of PMF values.
            save (bool): Whether to save the plot. Default is False.
            name (str): Filename for saving the plot. Default is 'polling_profile'.
            width (float): Width of the plot. Default is 0.5.
        """
        # Prepare multipanel plot
        fig = plt.figure(1, figsize=(10, 5))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=10, hspace=4.5)

        # Generate first panel
        ax = fig.add_subplot(gs[0:3, 0:5])
        ax.plot(delta_k_pmf - delta_k_pmf0, np.abs(pmf), linestyle='--', markerfacecolor='w', 
                markersize=8, color='r')
        ax.legend(loc='upper right')
        ax.set_xlabel(r'$\Delta k-\Delta k_0~(m^{-1})$')
        ax.set_ylabel(r'$10^{-3}$ $|$PMF$|$')

        if save:
            plt.savefig(f"{name}.png", dpi=300)
        plt.show()

    def plot_pmf_signal_idler(self, wavelength_signal: np.ndarray, pmf_signal: np.ndarray, 
                              pmf_signal_target: np.ndarray, wavelength_idler: np.ndarray, 
                              pmf_idler: np.ndarray, pmf_idler_target: np.ndarray, 
                              save: bool = False, name: str = 'polling_profile', width: float = 0.5) -> None:
        """
        Plots the PMF for both signal and idler wavelengths.

        Args:
            wavelength_signal (np.ndarray): Array of signal wavelengths.
            pmf_signal (np.ndarray): Calculated PMF for the signal.
            pmf_signal_target (np.ndarray): Target PMF for the signal.
            wavelength_idler (np.ndarray): Array of idler wavelengths.
            pmf_idler (np.ndarray): Calculated PMF for the idler.
            pmf_idler_target (np.ndarray): Target PMF for the idler.
            save (bool): Whether to save the plot. Default is False.
            name (str): Filename for saving the plot. Default is 'polling_profile'.
            width (float): Width of the plot. Default is 0.5.
        """
        # Prepare multipanel plot
        fig = plt.figure(1, figsize=(12, 5))
        gs = gridspec.GridSpec(5, 10)
        gs.update(wspace=10, hspace=4.5)

        # Normalize the PMF
        pmf_signal = pmf_signal / np.linalg.norm(np.abs(pmf_signal))
        pmf_idler = pmf_idler / np.linalg.norm(np.abs(pmf_idler))

        # Plot the PMF for the signal
        ax1 = fig.add_subplot(gs[0:5, 0:5])
        ax1.plot(wavelength_signal * 1e9, np.abs(pmf_signal_target), linestyle='-', markerfacecolor='w', 
                 markersize=8, color='k', label=r'Target')
        ax1.plot(wavelength_signal * 1e9, np.real(pmf_signal), linestyle='--', markerfacecolor='w', 
                 markersize=8, color='r', label=r'Custom - real')
        ax1.plot(wavelength_signal * 1e9, np.imag(pmf_signal), linestyle='--', markerfacecolor='w', 
                 markersize=8, color='g', label=r'Custom - imag')
        ax1.plot(wavelength_signal * 1e9, np.abs(pmf_signal), linestyle='--', markerfacecolor='w', 
                 markersize=8, color='b', label=r'Custom - abs')
        ax1.set_title('Signal')
        ax1.legend(loc='upper right')
        ax1.set_xlabel(r'Wavelength ($nm$)')
        ax1.set_ylabel(r'Normalized PMF')

        # Plot the PMF for the idler
        ax2 = fig.add_subplot(gs[0:5, 5:10])
        ax2.plot(wavelength_idler * 1e9, np.abs(pmf_idler_target), linestyle='-', markerfacecolor='w', 
                 markersize=8, color='k', label=r'Target')
        ax2.plot(wavelength_idler * 1e9, np.real(pmf_idler), linestyle='--', markerfacecolor='w', 
                 markersize=8, color='r', label=r'Custom - real')
        ax2.plot(wavelength_idler * 1e9, np.imag(pmf_idler), linestyle='--', markerfacecolor='w', 
                 markersize=8, color='g', label=r'Custom - imag')
        ax2.plot(wavelength_idler * 1e9, np.abs(pmf_idler), linestyle='--', markerfacecolor='w', 
                 markersize=8, color='b', label=r'Custom - abs')
        ax2.set_title('Idler')
        ax2.legend(loc='upper right')
        ax2.set_xlabel(r'Wavelength ($nm$)')
        ax2.set_ylabel(r'Normalized PMF')

        if save:
            plt.savefig(f"{name}.png", dpi=300)
        plt.show()

    def plot_jsa(self, jsa: np.ndarray, spectral_pmf_matrix: np.ndarray, 
                 spectral_pump_matrix: np.ndarray, signal_wavelength_array: np.ndarray, 
                 idler_wavelength_array: np.ndarray) -> None:
        """
        Plots the Joint Spectral Amplitude (JSA) and its components.

        Args:
            jsa (np.ndarray): Joint Spectral Amplitude matrix.
            spectral_pmf_matrix (np.ndarray): Spectral PMF matrix.
            spectral_pump_matrix (np.ndarray): Spectral pump matrix.
            signal_wavelength_array (np.ndarray): Array of signal wavelengths in meters.
            idler_wavelength_array (np.ndarray): Array of idler wavelengths in meters.
        """
        colormp = sns.color_palette("rocket", as_cmap=True)

        # Prepare multipanel plot
        fig = plt.figure(2, figsize=(10, 11))
        gs = gridspec.GridSpec(6, 6)
        gs.update(wspace=0.5, hspace=0.2)

        # Plot the JSA
        ax1 = fig.add_subplot(gs[2:6, 0:6])
        ax1.imshow(np.abs(jsa), origin='lower', extent=[signal_wavelength_array[0], signal_wavelength_array[-1], 
                                                        idler_wavelength_array[0], idler_wavelength_array[-1]], 
                   cmap=colormp)
        ax1.set_xlabel(r'$\lambda_i$ [nm]')
        ax1.set_ylabel(r'$\lambda_s$ [nm]')

        # Plot the spectral PMF
        ax2 = fig.add_subplot(gs[0:2, 0:2])
        ax2.imshow(np.abs(spectral_pmf_matrix), origin='lower', extent=[signal_wavelength_array[0], 
                                                                         signal_wavelength_array[-1], 
                                                                         idler_wavelength_array[0], 
                                                                         idler_wavelength_array[-1]], cmap=colormp)
        ax2.set_xlabel(r'$\lambda_i$ [nm]')
        ax2.set_ylabel(r'$\lambda_s$ [nm]')

        # Plot the spectral pump
        ax3 = fig.add_subplot(gs[0:2, 4:6])
        ax3.imshow(np.abs(spectral_pump_matrix), origin='lower', extent=[signal_wavelength_array[0], 
                                                                         signal_wavelength_array[-1], 
                                                                         idler_wavelength_array[0], 
                                                                         idler_wavelength_array[-1]], cmap=colormp)
        ax3.set_xlabel(r'$\lambda_i$ [nm]')
        ax3.set_ylabel(r'$\lambda_s$ [nm]')

        plt.tight_layout()
        plt.show()

    def plot_ga_metrics(self, metrics: dict) -> None:
        """
        Plots the metrics from a genetic algorithm (GA) run.

        Args:
            metrics (dict): Dictionary containing GA metrics such as 'best_fitness', 
                            'average_fitness', 'worst_fitness', 'std_dev_fitness', and 'generations'.
        """
        generations = range(metrics['generations'])
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(generations, metrics['best_fitness'], label=r'Best Fitness', color='k')
        plt.xlabel(r'Generation')
        plt.ylabel(r'Best Fitness')
        plt.title(r'Best Fitness Over Generations', color='r')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(generations, metrics['average_fitness'], label=r'Average Fitness', color='b')
        plt.xlabel(r'Generation')
        plt.ylabel(r'Average Fitness')
        plt.title(r'Average Fitness Over Generations')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(generations, metrics['worst_fitness'], label=r'Worst Fitness', color='g')
        plt.xlabel(r'Generation')
        plt.ylabel(r'Worst Fitness')
        plt.title(r'Worst Fitness Over Generations')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(generations, metrics['std_dev_fitness'], label=r'Std Dev of Fitness')
        plt.xlabel(r'Generation')
        plt.ylabel(r'Std Dev of Fitness')
        plt.title(r'Standard Deviation of Fitness Over Generations')
        plt.legend()

        plt.tight_layout()
        plt.show()