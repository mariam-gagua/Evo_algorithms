import numpy as np
import matplotlib.pyplot as plt
from evolutionary_optimization import EvolutionaryOptimizer
from pid import Twiddle
import time


def run_experiment(selection_sizes=[2, 5, 10, 20], n_trials=10):
    """Compare selection sizes with and without elitism"""
    with_elitism = []
    without_elitism = []

    for size in selection_sizes:
        with_e = []
        without_e = []
        for _ in range(n_trials):
            # With elitism
            optimizer = EvolutionaryOptimizer(selection_size=size, use_elitism=True)
            _, gens_with = optimizer.optimize()
            with_e.append(gens_with)

            # Without elitism
            optimizer = EvolutionaryOptimizer(selection_size=size, use_elitism=False)
            _, gens_without = optimizer.optimize()
            without_e.append(gens_without)

        with_elitism.append(np.mean(with_e))
        without_elitism.append(np.mean(without_e))

    return selection_sizes, with_elitism, without_elitism


def compare_mutation_rates(mutation_rates=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]):
    """Compare different mutation rates"""
    generations = []
    for rate in mutation_rates:
        optimizer = EvolutionaryOptimizer(mutation_std=rate)
        _, gens = optimizer.optimize()
        generations.append(gens)
    return mutation_rates, generations


def compare_selection_methods():
    """Compare all selection methods and Twiddle"""
    methods = ["truncation", "roulette", "tournament"]
    wall_times = []

    # Test evolutionary methods
    for method in methods:
        start_time = time.time()
        optimizer = EvolutionaryOptimizer()
        optimizer.optimize(selection_method=method)
        wall_times.append(time.time() - start_time)

    # Test Twiddle
    start_time = time.time()
    twiddle = Twiddle()
    # Pass the evaluate_controller function to run_iteration
    from main import evaluate_controller

    twiddle.run_iteration(evaluate_controller)
    wall_times.append(time.time() - start_time)
    methods.append("twiddle")

    return methods, wall_times


def plot_results():
    # Selection size comparison
    sizes, with_e, without_e = run_experiment()
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, with_e, "b-", label="With Elitism")
    plt.plot(sizes, without_e, "r--", label="Without Elitism")
    plt.xlabel("Selection Size")
    plt.ylabel("Generations to Converge")
    plt.title("Effect of Selection Size on Convergence")
    plt.legend()
    plt.savefig("selection_comparison.png")
    plt.close()

    # Mutation rate comparison
    rates, gens = compare_mutation_rates()
    plt.figure(figsize=(10, 5))
    plt.plot(rates, gens)
    plt.xlabel("Mutation Rate")
    plt.ylabel("Generations to Converge")
    plt.title("Effect of Mutation Rate on Convergence")
    plt.savefig("mutation_comparison.png")
    plt.close()

    # Selection methods comparison
    methods, times = compare_selection_methods()
    plt.figure(figsize=(10, 5))
    plt.bar(methods, times)
    plt.xlabel("Selection Method")
    plt.ylabel("Wall Time (seconds)")
    plt.title("Optimization Method Wall Time Comparison")
    plt.savefig("walltime_comparison.png")
    plt.close()


if __name__ == "__main__":
    plot_results()
