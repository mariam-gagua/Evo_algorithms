import numpy as np
import matplotlib.pyplot as plt
from evolutionary_optimization import EvolutionaryOptimizer
from pid import Twiddle
from main import evaluate_controller
import time


def optimize_evo(
    selection_method="truncation", selection_size=10, use_elitism=True, mutation_std=0.1
):
    """Run evolutionary optimization with specified parameters"""
    optimizer = EvolutionaryOptimizer(
        population_size=100,  # Fixed as per requirements
        selection_size=selection_size,
        use_elitism=use_elitism,
        mutation_std=mutation_std,
    )
    best_individual, history = optimizer.optimize(selection_method=selection_method)
    return best_individual, len(history)  # Return generations needed


def optimize_twiddle():
    """Run Twiddle optimization"""
    twiddle = Twiddle()
    best_error = float("inf")
    iterations = 0
    max_iterations = 100

    while iterations < max_iterations:
        params, error = twiddle.run_iteration(evaluate_controller)
        if error < best_error:
            best_error = error
        iterations += 1

    return params, iterations


def compare_selection_sizes():
    """Compare different selection sizes with and without elitism"""
    selection_sizes = [2, 5, 10, 20]
    generations_with_elitism = []
    generations_without_elitism = []
    std_with_elitism = []
    std_without_elitism = []

    n_trials = 10
    for size in selection_sizes:
        print(f"\nTesting selection size {size}")

        # With elitism
        gens_with = []
        for i in range(n_trials):
            print(f"Trial {i+1} with elitism")
            _, generations = optimize_evo(selection_size=size, use_elitism=True)
            gens_with.append(generations)
        generations_with_elitism.append(np.mean(gens_with))
        std_with_elitism.append(np.std(gens_with))

        # Without elitism
        gens_without = []
        for i in range(n_trials):
            print(f"Trial {i+1} without elitism")
            _, generations = optimize_evo(selection_size=size, use_elitism=False)
            gens_without.append(generations)
        generations_without_elitism.append(np.mean(gens_without))
        std_without_elitism.append(np.std(gens_without))

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        selection_sizes,
        generations_with_elitism,
        yerr=std_with_elitism,
        label="With Elitism",
        color="blue",
        capsize=5,
    )
    plt.errorbar(
        selection_sizes,
        generations_without_elitism,
        yerr=std_without_elitism,
        label="Without Elitism",
        color="red",
        capsize=5,
    )
    plt.xlabel("Selection Size")
    plt.ylabel("Mean Generations to Converge")
    plt.title("Effect of Selection Size on Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig("selection_size_comparison.png")
    plt.close()


def compare_mutation_rates():
    """Compare different mutation rates"""
    mutation_stds = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
    generations_mean = []
    generations_std = []

    n_trials = 10
    for std in mutation_stds:
        print(f"\nTesting mutation std {std}")
        gens = []
        for i in range(n_trials):
            print(f"Trial {i+1}")
            _, generations = optimize_evo(mutation_std=std)
            gens.append(generations)
        generations_mean.append(np.mean(gens))
        generations_std.append(np.std(gens))

    plt.figure(figsize=(10, 6))
    plt.errorbar(mutation_stds, generations_mean, yerr=generations_std, capsize=5)
    plt.xlabel("Mutation Standard Deviation")
    plt.ylabel("Mean Generations to Converge")
    plt.title("Effect of Mutation Rate on Convergence")
    plt.grid(True)
    plt.savefig("mutation_rate_comparison.png")
    plt.close()


def compare_selection_methods():
    """Compare different selection methods and Twiddle"""
    methods = ["truncation", "roulette", "tournament"]
    generations_mean = []
    generations_std = []
    wall_times_mean = []
    wall_times_std = []

    n_trials = 10
    for method in methods:
        print(f"\nTesting {method} selection")
        gens = []
        times = []
        for i in range(n_trials):
            print(f"Trial {i+1}")
            start_time = time.time()
            _, generations = optimize_evo(selection_method=method)
            end_time = time.time()
            gens.append(generations)
            times.append(end_time - start_time)
        generations_mean.append(np.mean(gens))
        generations_std.append(np.std(gens))
        wall_times_mean.append(np.mean(times))
        wall_times_std.append(np.std(times))

    # Add Twiddle comparison
    print("\nTesting Twiddle")
    twiddle_times = []
    for i in range(n_trials):
        print(f"Trial {i+1}")
        start_time = time.time()
        optimize_twiddle()
        end_time = time.time()
        twiddle_times.append(end_time - start_time)
    wall_times_mean.append(np.mean(twiddle_times))
    wall_times_std.append(np.std(twiddle_times))

    # Plot wall time comparison
    plt.figure(figsize=(10, 6))
    methods.append("Twiddle")
    plt.bar(methods, wall_times_mean, yerr=wall_times_std, capsize=5)
    plt.xlabel("Optimization Method")
    plt.ylabel("Wall Time (seconds)")
    plt.title("Optimization Method Wall Time Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("wall_time_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("Comparing selection sizes...")
    compare_selection_sizes()

    print("\nComparing mutation rates...")
    compare_mutation_rates()

    print("\nComparing selection methods...")
    compare_selection_methods()
