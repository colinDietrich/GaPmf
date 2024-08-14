from random import choices, randint, randrange, random, shuffle, sample
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Callable, Optional
from IPython.display import clear_output
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from GaPmf.medium.Crystal import Crystal

class GA:
    def __init__(self, size: int, option: str, Ps: float, Pu1: float, Pu2: float,
                 weight_Pm: float, mode: int, domain_width: float,
                 max_length: float, min_length: float,
                 phase_mismatch_array: Optional[np.ndarray] = None, 
                 pmf_target: Optional[np.ndarray] = None,
                 signal_phase_mismatch_array: Optional[np.ndarray] = None,
                 signal_wavelength_array: Optional[np.ndarray] = None, 
                 idler_wavelength_array: Optional[np.ndarray] = None,
                 idler_phase_mismatch_array: Optional[np.ndarray] = None, 
                 signal_pmf_target: Optional[np.ndarray] = None,
                 idler_pmf_target: Optional[np.ndarray] = None):
        """
        Initializes the Genetic Algorithm (GA) with the given parameters.
        
        Args:
            size (int): Population size.
            option (str): Option for APLN initialization.
            Ps (float): Crossover probability for single-point crossover.
            Pu1 (float): Crossover probability for uniform crossover.
            Pu2 (float): Probability for each gene in uniform crossover.
            weight_Pm (float): Mutation weight.
            mode (int): Mode of operation (1 or 2).
            domain_width (float): Domain width for APLN.
            max_length (float): Maximum length for crystal domains.
            min_length (float): Minimum length for crystal domains.
            phase_mismatch_array (Optional[np.ndarray]): Phase mismatch array (required for mode 1).
            pmf_target (Optional[np.ndarray]): Target phase matching function (PMF) (required for mode 1).
            signal_phase_mismatch_array (Optional[np.ndarray]): Signal phase mismatch array (required for mode 2).
            signal_wavelength_array (Optional[np.ndarray]): Signal wavelength array (required for mode 2).
            idler_wavelength_array (Optional[np.ndarray]): Idler wavelength array (required for mode 2).
            idler_phase_mismatch_array (Optional[np.ndarray]): Idler phase mismatch array (required for mode 2).
            signal_pmf_target (Optional[np.ndarray]): Target signal PMF (required for mode 2).
            idler_pmf_target (Optional[np.ndarray]): Target idler PMF (required for mode 2).
        """
        self.size = size
        self.option = option
        self.Ps = Ps
        self.Pu1 = Pu1
        self.Pu2 = Pu2
        self.weight_Pm = weight_Pm
        self.domain_width = domain_width
        self.max_length = max_length
        self.min_length = min_length
        self.mode = mode

        # Validate mode and required arrays
        if self.mode == 1:
            if phase_mismatch_array is not None and pmf_target is not None:
                self.phase_mismatch_array = phase_mismatch_array
                self.pmf_target = pmf_target
            else:
                raise ValueError('Phase mismatch array and PMF target must be provided for mode 1.')
        elif self.mode == 2:
            if (signal_wavelength_array is not None and idler_wavelength_array is not None and
                signal_phase_mismatch_array is not None and idler_phase_mismatch_array is not None and
                signal_pmf_target is not None and idler_pmf_target is not None):
                self.phase_mismatch_array = phase_mismatch_array
                self.pmf_target = pmf_target
                self.signal_wavelength_array = signal_wavelength_array
                self.idler_wavelength_array = idler_wavelength_array
                self.signal_phase_mismatch_array = signal_phase_mismatch_array
                self.idler_phase_mismatch_array = idler_phase_mismatch_array
                self.signal_pmf_target = signal_pmf_target
                self.idler_pmf_target = idler_pmf_target
            else:
                raise ValueError('All required arrays must be provided for mode 2.')
        else:
            raise ValueError('Mode must be 1 or 2.')

    def generate_genome(self, option: str) -> 'Crystal':
        """
        Generates a new crystal genome using the APLN class.
        
        Args:
            option (str): Option for genome generation.
        
        Returns:
            Crystal: A newly generated crystal genome.
        """
        if self.mode == 1:
            return Crystal(self.domain_width, option, self.max_length, self.min_length,
                           self.mode, phase_mismatch_array=self.phase_mismatch_array,
                           pmf_target=self.pmf_target)
        else:
            return Crystal(self.domain_width, option, self.max_length, self.min_length,
                           self.mode, signal_phase_mismatch_array=self.signal_phase_mismatch_array,
                           idler_phase_mismatch_array=self.idler_phase_mismatch_array,
                           signal_pmf_target=self.signal_pmf_target,
                           idler_pmf_target=self.idler_pmf_target)

    def generate_population(self, size: int, option: str) -> List['Crystal']:
        """
        Generates the initial population of genomes.
        
        Args:
            size (int): The size of the population.
            option (str): Option for genome generation.
        
        Returns:
            List[Crystal]: A list of generated genomes.
        """
        return [self.generate_genome(option) for _ in range(size)]

    def fitness_function(self, genome: 'Crystal') -> float:
        """
        Calculates the fitness of a genome based on the weighted MSE of real and imaginary parts.
        
        Args:
            genome (Crystal): The genome to evaluate.
        
        Returns:
            float: The fitness value of the genome.
        """
        if self.mode == 1:
            return -genome.mse_abs
        else:
            return -genome.mse_abs_signal - genome.mse_abs_idler

    def single_point_crossover(self, a: 'Crystal', b: 'Crystal') -> Tuple['Crystal', 'Crystal']:
        """
        Performs single-point crossover on two genomes.
        
        Args:
            a (Crystal): The first parent genome.
            b (Crystal): The second parent genome.
        
        Returns:
            Tuple[Crystal, Crystal]: The resulting pair of genomes after crossover.
        """
        length = min(len(a.domain_values), len(b.domain_values))
        if length < 2:
            return a, b
        
        p = randint(1, length - 1)
        saveA = a.domain_values
        saveB = b.domain_values
        
        a.domain_values = saveA[:p] + saveB[p:]
        b.domain_values = saveB[:p] + saveA[p:]
        
        if a.length > self.max_length or a.length < self.min_length:
            a.domain_values = saveA
        if b.length > self.max_length or b.length < self.min_length:
            b.domain_values = saveB
        
        return a, b

    def uniform_crossover(self, a: 'Crystal', b: 'Crystal') -> Tuple['Crystal', 'Crystal']:
        """
        Performs uniform crossover on two genomes.
        
        Args:
            a (Crystal): The first parent genome.
            b (Crystal): The second parent genome.
        
        Returns:
            Tuple[Crystal, Crystal]: The resulting pair of genomes after crossover.
        """
        la = len(a.domain_values)
        lb = len(b.domain_values)
        minL = min(la, lb)
        offset_a = la - lb if la > lb else 0
        offset_b = lb - la if lb > la else 0
        
        i_new = 0
        i_old = 0
        save_a = a.domain_values[:offset_a]
        save_b = b.domain_values[:offset_b]
        
        while i_old < minL:
            up = minL - i_old
            i_new += randint(1, up)
            if random() < self.Pu2:
                save_a += b.domain_values[offset_b + i_old: offset_b + i_new]
                save_b += a.domain_values[offset_a + i_old: offset_a + i_new]
            else:
                save_a += a.domain_values[offset_a + i_old: offset_a + i_new]
                save_b += b.domain_values[offset_b + i_old: offset_b + i_new]
            i_old = i_new
        
        a.domain_values = save_a + a.domain_values[offset_a + i_old:]
        b.domain_values = save_b + b.domain_values[offset_b + i_old:]
        
        return a, b

    def mutation(self, genome: 'Crystal', num: int = 1) -> 'Crystal':
        """
        Mutates a genome with a probability inversely proportional to its length.
        
        Args:
            genome (Crystal): The genome to mutate.
            num (int): The number of mutations to perform (default is 1).
        
        Returns:
            Crystal: The mutated genome.
        """
        Pm = 1 / len(genome.domain_values) * self.weight_Pm
        for _ in range(num):
            index = randrange(len(genome.domain_values))
            if random() <= Pm:
                genome.domain_values[index] *= -1
        return genome

    def population_fitness(self, population: List['Crystal'], fitness_func: Callable[['Crystal'], float]) -> Tuple[float, List[float]]:
        """
        Calculates the total fitness of a population.
        
        Args:
            population (List[Crystal]): The population to evaluate.
            fitness_func (Callable[[Crystal], float]): The fitness function.
        
        Returns:
            Tuple[float, List[float]]: The total fitness and a list of individual fitness values.
        """
        with ProcessPoolExecutor() as executor:
            fitness_values = list(executor.map(fitness_func, population))
        return sum(fitness_values), fitness_values

    def selection_pair(self, population: List['Crystal'], fitness_func: Callable[['Crystal'], float]) -> Tuple['Crystal', 'Crystal']:
        """
        Selects a pair of genomes from the population based on their fitness.
        
        Args:
            population (List[Crystal]): The population to select from.
            fitness_func (Callable[[Crystal], float]): The fitness function.
        
        Returns:
            Tuple[Crystal, Crystal]: A pair of selected genomes.
        """
        return tuple(choices(population=population,
                             weights=[fitness_func(gene) for gene in population],
                             k=2))

    def sort_population(self, population: List['Crystal'], fitness_values: List[float]) -> List['Crystal']:
        """
        Sorts the population based on their fitness in descending order.
        
        Args:
            population (List[Crystal]): The population to sort.
            fitness_values (List[float]): The fitness values associated with the population.
        
        Returns:
            List[Crystal]: The sorted population.
        """
        return [x for _, x in sorted(zip(fitness_values, population), key=lambda pair: pair[0], reverse=True)]

    def print_stats(self, population: List['Crystal'], generation_id: int, metrics: dict) -> None:
        """
        Prints statistics about the current generation.
        
        Args:
            population (List[Crystal]): The current population.
            generation_id (int): The ID of the current generation.
            metrics (dict): Dictionary containing metrics for the population.
        """
        clear_output(wait=True)
        
        print(f"GENERATION {generation_id:02d}")
        print(f"\t - Population length: {len(population)}")
        print(f"\t - Avg. Fitness: {metrics['average_fitness'][generation_id]:.6f}")
        print(f"\t - Avg. Length: {metrics['average_length'][generation_id]:.6f}")
        print(f"\t - Best Crystal")
        print(f"\t\t -> length: {metrics['best_length'][generation_id]}")
        print(f"\t\t -> fitness: ({metrics['best_fitness'][generation_id]:.6f})")
        print(f"\t - Worst Crystal")
        print(f"\t\t -> length: {metrics['worst_length'][generation_id]}")
        print(f"\t\t -> fitness: ({metrics['worst_fitness'][generation_id]:.6f})")

        best_genome = population[0]

        fig, axs = plt.subplots(2, 2, figsize=(12, 6))

        if self.mode == 1:
            # Plot the target and simulated PMF profiles
            axs[0, 0].plot(self.phase_mismatch_array, np.abs(self.pmf_target), '--', label='Target', color='black')
            axs[0, 0].plot(self.phase_mismatch_array, np.abs(best_genome.pmf) / np.linalg.norm(np.abs(best_genome.pmf)), label='Simulated', color='crimson')
            axs[0, 0].legend()
            axs[0, 0].set_title('PMF Profiles')
            axs[0, 0].set_xlabel(r'$\Delta k ($m^{-1}$)$')
            axs[0, 0].set_ylabel('PMF Amplitude (a.u.)')

            # Plot best fitness over generations
            axs[0, 1].plot(metrics['best_fitness'], label='Best Fitness', color='forestgreen')
            axs[0, 1].legend()
            axs[0, 1].set_title('Best Fitness Over Generations')
            axs[0, 1].set_xlabel('Generation')
            axs[0, 1].set_ylabel('Best Fitness')
        else:
            # Plot the target and simulated PMF profiles for the signal
            axs[0, 0].plot(self.signal_wavelength_array, np.abs(self.signal_pmf_target), '--', label='Target Signal', color='black')
            axs[0, 0].plot(self.signal_wavelength_array, np.abs(best_genome.pmf_signal) / np.linalg.norm(np.abs(best_genome.pmf_signal)), label='Simulated Signal', color='crimson')
            axs[0, 0].legend()
            axs[0, 0].set_title('Signal PMF Profiles')
            axs[0, 0].set_xlabel('Wavelength (nm)')
            axs[0, 0].set_ylabel('PMF Amplitude')

            # Plot the target and simulated PMF profiles for the idler
            axs[0, 1].plot(self.idler_wavelength_array, np.abs(self.idler_pmf_target), '--', label='Target Idler', color='royalblue')
            axs[0, 1].plot(self.idler_wavelength_array, np.abs(best_genome.pmf_idler) / np.linalg.norm(np.abs(best_genome.pmf_idler)), label='Simulated Idler', color='forestgreen')
            axs[0, 1].legend()
            axs[0, 1].set_title('Idler PMF Profiles')
            axs[0, 1].set_xlabel('Wavelength (nm)')
            axs[0, 1].set_ylabel('PMF Amplitude')

        axs[1, 0].plot(metrics['average_fitness'], label='Average Fitness', color='b')
        axs[1, 0].legend()
        axs[1, 0].set_xlabel('Generation')
        axs[1, 0].set_ylabel('Average Fitness')
        axs[1, 0].set_title('Average Fitness Over Generations')

        axs[1, 1].plot(metrics['std_dev_fitness'], label='Std Dev of Fitness', color='r')
        axs[1, 1].legend()
        axs[1, 1].set_xlabel('Generation')
        axs[1, 1].set_ylabel('Std Dev of Fitness')
        axs[1, 1].set_title('Standard Deviation of Fitness Over Generations')

        plt.tight_layout()
        plt.show()

    def run_evolution(self, nb_generation: int, printer: bool = False, fitness_limit: float = 0, nb_level: int = 1, 
                      restart: Optional[int] = None, restart_depth: int = 4, population: Optional[List['Crystal']] = None) -> Tuple[List['Crystal'], int]:
        """
        Runs the genetic algorithm for a given number of generations.
        
        Args:
            nb_generation (int): Number of generations to run.
            printer (bool): Whether to print statistics during evolution.
            fitness_limit (float): Fitness limit to stop the evolution early.
            nb_level (int): Level for domain width adjustment.
            restart (Optional[int]): Generation interval for restarting the population.
            restart_depth (int): Number of new genomes to inject during restart.
            population (Optional[List[Crystal]]): Initial population. If None, a new population will be generated.
        
        Returns:
            Tuple[List[Crystal], int]: The final population and the number of generations completed.
        """
        if population is None:
            population = self.generate_population(self.size, self.option)
        else:
            total_fitness, fitness_values = self.population_fitness(population, self.fitness_function)
            population = self.sort_population(population, fitness_values)
        
        self.metrics = {
            'best_fitness': [],
            'best_length': [],
            'average_fitness': [],
            'average_length': [],
            'worst_fitness': [],
            'worst_length': [],
            'std_dev_fitness': [],
            'generations': 0
        }
        level = 1

        for i in tqdm(range(nb_generation), desc="Evolution Progress"):
            start_time = time.time()
            total_fitness, fitness_values = self.population_fitness(population, self.fitness_function)
            self.metrics['best_fitness'].append(max(fitness_values))
            self.metrics['average_fitness'].append(np.mean(fitness_values))
            self.metrics['worst_fitness'].append(min(fitness_values))
            self.metrics['best_length'].append(population[0].length)
            self.metrics['average_length'].append(np.mean([genome.length for genome in population]))
            self.metrics['worst_length'].append(population[-1].length)
            self.metrics['std_dev_fitness'].append(np.std(fitness_values))
            self.metrics['generations'] += 1

            if printer:
                self.print_stats(population, i, self.metrics)

            next_generation = []

            if i in nb_level:
                level += 1
                self.domain_width /= 2

            # Crossover and mutation
            best_genomes = population[:self.size // 10]
            shuffle(population)

            with ProcessPoolExecutor() as executor:
                futures = []
                for j in range(len(population) // 2):
                    parents = [population[2 * j], population[2 * j + 1]]
                    if random() < self.Pu1:
                        parents[0], parents[1] = self.uniform_crossover(parents[0], parents[1])
                    if random() < self.Ps:
                        parents[0], parents[1] = self.single_point_crossover(parents[0], parents[1])
                    futures.append(executor.submit(self.mutation, parents[0]))
                    futures.append(executor.submit(self.mutation, parents[1]))
                for future in as_completed(futures):
                    future.result().update(level=level)
                    next_generation.append(future.result())

            new_population_injection = self.generate_population(self.size // 10, 'random')
            N = self.size - len(best_genomes) - len(new_population_injection)
            population = best_genomes + new_population_injection + sample(next_generation, N)
            total_fitness, fitness_values = self.population_fitness(population, self.fitness_function)
            population = self.sort_population(population, fitness_values)

            if restart is not None:
                if i % restart == 0 and i != 0:
                    new_population_injection = self.generate_population(restart_depth, 'random')
                    population = new_population_injection + population[:self.size - restart_depth]
                    total_fitness, fitness_values = self.population_fitness(population, self.fitness_function)
                    population = self.sort_population(population, fitness_values)

            end_time = time.time()

        return population, i