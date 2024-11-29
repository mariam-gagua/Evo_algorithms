import numpy as np
import matplotlib.pyplot as plt
from evolutionary_optimization import EvolutionaryOptimizer
from pid import Twiddle
import time


def run_experiment(selection_method, selection_sizes=[2, 5, 10, 20], n_trials=5):
    results = {"with_elitism": [], "without_elitism": []}

    print(f"\n{'='*50}")
    print(f"Testing {selection_method} selection")
    print(f"{'='*50}\n")

    for selection_size in selection_sizes:
        print(f"Testing {selection_method} selection with size {selection_size}")
        for use_elitism in [True, False]:
            generations_list = []
            for trial in range(n_trials):
                print(f"Progress: {trial + 1}/{n_trials}")
                optimizer = EvolutionaryOptimizer(
                    selection_size=selection_size, use_elitism=use_elitism
                )
                _, generations = optimizer.optimize(selection_method=selection_method)
                generations_list.append(
                    generations
                )  # generations is already an integer

            key = "with_elitism" if use_elitism else "without_elitism"
            results[key].append(np.mean(generations_list))

    return results


def compare_methods():
    methods = ["truncation", "roulette", "tournament"]
    all_results = {}
    method_times = {}
    method_time_stds = {}

    for method in methods:
        print(f"\nTesting {method} selection method")
        # Single run of experiment with timing
        start_time = time.time()
        results = run_experiment(method)
        end_time = time.time()

        method_times[method] = end_time - start_time
        method_time_stds[method] = 0
        all_results[method] = results

    # Add Twiddle timing
    start_time = time.time()
    twiddle = Twiddle()
    from main import evaluate_controller

    twiddle.run_iteration(evaluate_controller)
    method_times["twiddle"] = time.time() - start_time
    method_time_stds["twiddle"] = 0

    # Add print statements for timing results
    print("\nWall Time Results:")
    print("=" * 40)
    for method, time_taken in method_times.items():
        print(f"{method.capitalize():12} : {time_taken:.3f} seconds")
    print("=" * 40)

    # Plot final results
    plot_results(all_results, method_times, method_time_stds, "final_results.png")

    return all_results, method_times, method_time_stds


def plot_results(all_results, method_times, method_time_stds, filename="results.png"):
    selection_sizes = [2, 5, 10, 20]
    colors = ["b", "g", "r"]

    # Plot selection size comparison
    plt.figure(figsize=(10, 5))
    for i, (method, results) in enumerate(all_results.items()):
        plt.plot(
            selection_sizes,
            results["with_elitism"],
            f"{colors[i]}-",
            label=f"{method} with elitism",
        )
        plt.plot(
            selection_sizes,
            results["without_elitism"],
            f"{colors[i]}--",
            label=f"{method} without elitism",
        )

    plt.xlabel("Selection Size")
    plt.ylabel("Generations to Converge")
    plt.title("Effect of Selection Size on Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

    # Modified wall time comparison plot
    plt.figure(figsize=(10, 5))
    methods = list(method_times.keys())
    times = [method_times[m] for m in methods]
    time_stds = [method_time_stds[m] for m in methods]

    bars = plt.bar(methods, times, yerr=time_stds, capsize=5)
    plt.xlabel("Selection Method")
    plt.ylabel("Wall Time (seconds)")
    plt.title("Optimization Method Wall Time Comparison")
    plt.xticks(rotation=45)
    plt.grid(True)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("wall_time_comparison.png")
    plt.close()


if __name__ == "__main__":
    all_results, method_times, method_time_stds = compare_methods()
