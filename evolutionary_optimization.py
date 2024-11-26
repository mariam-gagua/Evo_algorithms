import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from main import evaluate_controller
import time


class Individual:
    def __init__(self, genes):
        self.genes = genes  # [P, I, D]
        self.fitness = None

    def evaluate(self):
        if self.fitness is None:
            error = evaluate_controller(self.genes)
            self.fitness = -error  # negative because we want to maximize fitness
        return self.fitness


class EvolutionaryOptimizer:
    def __init__(
        self, population_size=100, mutation_std=0.1, selection_size=10, use_elitism=True
    ):
        self.population_size = population_size
        self.mutation_std = mutation_std
        self.selection_size = selection_size
        self.use_elitism = use_elitism

        # init param ranges (based on Twiddle)
        self.p_range = (0, 0.2)
        self.i_range = (0, 0.005)
        self.d_range = (0, 7.0)

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            genes = np.array(
                [
                    np.random.uniform(*self.p_range),
                    np.random.uniform(*self.i_range),
                    np.random.uniform(*self.d_range),
                ]
            )
            population.append(Individual(genes))
        return population

    def mutate(self, individual):
        new_genes = individual.genes + np.random.normal(0, self.mutation_std, size=3)
        # clip to valid ranges
        new_genes = np.clip(
            new_genes,
            [self.p_range[0], self.i_range[0], self.d_range[0]],
            [self.p_range[1], self.i_range[1], self.d_range[1]],
        )
        return Individual(new_genes)

    def truncation_selection(self, population):
        sorted_pop = sorted(population, key=lambda x: x.evaluate(), reverse=True)
        return sorted_pop[: self.selection_size]

    def roulette_wheel_selection(self, population):
        fitnesses = np.array([ind.evaluate() for ind in population])
        min_fitness = min(fitnesses)
        fitnesses = fitnesses - min_fitness + 1e-6
        probabilities = fitnesses / sum(fitnesses)

        selected_indices = np.random.choice(
            len(population), size=self.selection_size, p=probabilities, replace=True
        )
        return [population[i] for i in selected_indices]

    def tournament_selection(self, population, tournament_size=5):
        selected = []
        for _ in range(self.selection_size):
            tournament = np.random.choice(
                population, size=tournament_size, replace=False
            )
            winner = max(tournament, key=lambda x: x.evaluate())
            selected.append(winner)
        return selected

    def optimize(self, max_generations=100, selection_method="truncation"):
        population = self.initialize_population()
        best_fitness_history = []
        best_individual = None
        generations_without_improvement = 0

        # selection method mapping
        selection_methods = {
            "truncation": self.truncation_selection,
            "roulette": self.roulette_wheel_selection,
            "tournament": self.tournament_selection,
        }
        select = selection_methods[selection_method]

        print(f"Starting optimization with {selection_method} selection...")

        for generation in range(max_generations):
            # evaluate population
            for ind in population:
                ind.evaluate()

            # find best individual
            current_best = max(population, key=lambda x: x.evaluate())
            if (
                best_individual is None
                or current_best.evaluate() > best_individual.evaluate()
            ):
                best_individual = Individual(current_best.genes.copy())
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # record best fitness
            best_fitness_history.append(
                -best_individual.evaluate()
            )  # convert back to error

            if generation % 10 == 0:
                print(
                    f"Generation {generation}: Best error = {-best_individual.evaluate():.2f}"
                )

            # check convergence
            if (
                generations_without_improvement >= 20
            ):  # no improvement for 20 generations
                print(f"Converged after {generation} generations")
                break

            # selection
            selected = select(population)

            # create new population
            new_population = []
            if self.use_elitism:
                new_population.append(best_individual)

            while len(new_population) < self.population_size:
                parent = np.random.choice(selected)
                child = self.mutate(parent)
                new_population.append(child)

            population = new_population

        return best_individual, best_fitness_history


def run_comparison(selection_sizes=[2, 5, 10, 20], n_trials=5):
    results = {"with_elitism": [], "without_elitism": []}

    for selection_size in selection_sizes:
        for use_elitism in [True, False]:
            convergence_gens = []
            for trial in range(n_trials):
                print(
                    f"\nTrial {trial + 1} with selection_size={selection_size}, elitism={use_elitism}"
                )
                optimizer = EvolutionaryOptimizer(
                    selection_size=selection_size, use_elitism=use_elitism
                )
                _, history = optimizer.optimize()
                convergence_gens.append(len(history))

            key = "with_elitism" if use_elitism else "without_elitism"
            results[key].append(np.mean(convergence_gens))

    return results


if __name__ == "__main__":
    optimizer = EvolutionaryOptimizer(selection_size=10, use_elitism=True)
    best_individual, history = optimizer.optimize()

    print("\nBest parameters found:")
    print(f"P: {best_individual.genes[0]:.6f}")
    print(f"I: {best_individual.genes[1]:.6f}")
    print(f"D: {best_individual.genes[2]:.6f}")
    print(f"Final error: {-best_individual.evaluate():.6f}")
