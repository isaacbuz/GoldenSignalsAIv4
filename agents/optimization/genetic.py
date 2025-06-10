# optimization/genetic.py
# Purpose: Implements a genetic algorithm to optimize trading strategy parameters,
# focusing on improving performance metrics (e.g., Sharpe ratio, drawdown) for options
# trading strategies.

import logging
import random
from typing import Dict, List

import numpy as np

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class GeneticOptimizer:
    """Uses genetic algorithms to optimize agent parameters for better performance."""

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 20,
        mutation_rate: float = 0.1,
    ):
        """Initialize the GeneticOptimizer.

        Args:
            population_size (int): Number of individuals in the population.
            generations (int): Number of generations to evolve.
            mutation_rate (float): Probability of mutation for each parameter.
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        logger.info(
            {
                "message": "GeneticOptimizer initialized",
                "population_size": population_size,
                "generations": generations,
                "mutation_rate": mutation_rate,
            }
        )

    def initialize_population(self, param_ranges: Dict) -> List[Dict]:
        """Initialize a population of parameter sets.

        Args:
            param_ranges (Dict): Parameter ranges (e.g., {'threshold': (0, 1)}).

        Returns:
            List[Dict]: List of parameter sets.
        """
        logger.debug({"message": "Initializing population"})
        try:
            population = []
            for _ in range(self.population_size):
                individual = {}
                for param, (min_val, max_val) in param_ranges.items():
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        individual[param] = random.randint(min_val, max_val)
                    else:
                        individual[param] = random.uniform(min_val, max_val)
                population.append(individual)
            logger.debug(
                {
                    "message": f"Population initialized with {len(population)} individuals"
                }
            )
            return population
        except Exception as e:
            logger.error({"message": f"Failed to initialize population: {str(e)}"})
            return []

    def evaluate_fitness(self, individual: Dict, performance_metrics: Dict) -> float:
        """Evaluate the fitness of an individual based on performance metrics.

        Args:
            individual (Dict): Parameter set to evaluate.
            performance_metrics (Dict): Metrics like 'total_return', 'sharpe_ratio', 'max_drawdown'.

        Returns:
            float: Fitness score (higher is better).
        """
        logger.debug({"message": f"Evaluating fitness for individual: {individual}"})
        try:
            sharpe_ratio = performance_metrics.get("sharpe_ratio", 0.0)
            max_drawdown = performance_metrics.get("max_drawdown", 0.0)
            # Fitness function: prioritize Sharpe ratio, penalize drawdown
            fitness = sharpe_ratio - 0.5 * abs(max_drawdown)
            logger.debug({"message": f"Fitness score: {fitness:.4f}"})
            return fitness
        except Exception as e:
            logger.error({"message": f"Failed to evaluate fitness: {str(e)}"})
            return 0.0

    def select_parents(
        self, population: List[Dict], fitness_scores: List[float]
    ) -> List[Dict]:
        """Select parents for crossover using tournament selection.

        Args:
            population (List[Dict]): List of parameter sets.
            fitness_scores (List[float]): Fitness scores for each individual.

        Returns:
            List[Dict]: Selected parents.
        """
        logger.debug({"message": "Selecting parents"})
        try:
            parents = []
            for _ in range(len(population) // 2):
                tournament = random.sample(range(len(population)), 3)
                winner = max(tournament, key=lambda i: fitness_scores[i])
                parents.append(population[winner])
            logger.debug({"message": f"Selected {len(parents)} parents"})
            return parents
        except Exception as e:
            logger.error({"message": f"Failed to select parents: {str(e)}"})
            return []

    def crossover(self, parent1: Dict, parent2: Dict) -> tuple:
        """Perform crossover between two parents to create children.

        Args:
            parent1 (Dict): First parent's parameter set.
            parent2 (Dict): Second parent's parameter set.

        Returns:
            tuple: Two children parameter sets.
        """
        logger.debug({"message": "Performing crossover"})
        try:
            child1, child2 = {}, {}
            for param in parent1:
                if random.random() < 0.5:
                    child1[param], child2[param] = parent1[param], parent2[param]
                else:
                    child1[param], child2[param] = parent2[param], parent1[param]
            return child1, child2
        except Exception as e:
            logger.error({"message": f"Failed to perform crossover: {str(e)}"})
            return {}, {}

    def mutate(self, individual: Dict, param_ranges: Dict):
        """Mutate an individual's parameters.

        Args:
            individual (Dict): Parameter set to mutate.
            param_ranges (Dict): Parameter ranges for mutation bounds.
        """
        logger.debug({"message": f"Mutating individual: {individual}"})
        try:
            for param, (min_val, max_val) in param_ranges.items():
                if random.random() < self.mutation_rate:
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        individual[param] = random.randint(min_val, max_val)
                    else:
                        individual[param] = random.uniform(min_val, max_val)
            logger.debug({"message": f"Mutated individual: {individual}"})
        except Exception as e:
            logger.error({"message": f"Failed to mutate individual: {str(e)}"})

    def optimize(self, param_ranges: Dict, performance_metrics: Dict) -> Dict:
        """Optimize agent parameters using a genetic algorithm.

        Args:
            param_ranges (Dict): Parameter ranges for optimization (e.g., {'threshold': (0, 1)}).
            performance_metrics (Dict): Metrics to evaluate fitness (e.g., {'sharpe_ratio': 1.5}).

        Returns:
            Dict: Best parameter set for options trading strategy.
        """
        logger.info({"message": "Starting genetic optimization"})
        try:
            population = self.initialize_population(param_ranges)
            if not population:
                logger.error({"message": "Population initialization failed"})
                return {}

            for generation in range(self.generations):
                # Evaluate fitness for all individuals
                fitness_scores = [
                    self.evaluate_fitness(ind, performance_metrics)
                    for ind in population
                ]

                # Select parents
                parents = self.select_parents(population, fitness_scores)

                # Create next generation
                next_gen = []
                for i in range(0, len(parents), 2):
                    if i + 1 < len(parents):
                        child1, child2 = self.crossover(parents[i], parents[i + 1])
                        self.mutate(child1, param_ranges)
                        self.mutate(child2, param_ranges)
                        next_gen.extend([child1, child2])
                population = next_gen + parents[: self.population_size - len(next_gen)]

                # Log best individual in this generation
                best_fitness = max(fitness_scores)
                best_individual = population[np.argmax(fitness_scores)]
                logger.info(
                    {
                        "message": f"Generation {generation+1}: Best fitness = {best_fitness:.4f}",
                        "best_individual": best_individual,
                    }
                )

            # Return the best individual after all generations
            final_fitness_scores = [
                self.evaluate_fitness(ind, performance_metrics) for ind in population
            ]
            best_individual = population[np.argmax(final_fitness_scores)]
            logger.info(
                {
                    "message": "Genetic optimization completed",
                    "best_parameters": best_individual,
                }
            )
            return best_individual
        except Exception as e:
            logger.error(
                {"message": f"Failed to perform genetic optimization: {str(e)}"}
            )
            return {}
