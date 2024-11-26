import numpy as np
import matplotlib.pyplot as plt
from evolutionary_optimization import EvolutionaryOptimizer
from main import evaluate_controller
import time


def run_experiment(
    selection_method,
    selection_sizes=[2, 5, 10, 20],
    mutation_stds=[0.01, 0.05, 0.1, 0.2],
    n_trials=5,
):
    # store results
    results = {
        "with_elitism": {size: [] for size in selection_sizes},
        "without_elitism": {size: [] for size in selection_sizes},
        "mutation_study": {mut: [] for mut in mutation_stds},
    }

    # selection sizes with and without elitism
    for selection_size in selection_sizes:
        print(f"\nTesting selection size {selection_size}")
        for use_elitism in [True, False]:
            generations_list = []
            for trial in range(n_trials):
                optimizer = EvolutionaryOptimizer(
                    selection_size=selection_size,
                    use_elitism=use_elitism,
                    mutation_std=0.1,  # default mutation rate
                )
                _, history = optimizer.optimize(selection_method=selection_method)
                generations_list.append(len(history))

            key = "with_elitism" if use_elitism else "without_elitism"
            results[key][selection_size] = np.mean(generations_list)

    # mutation rates (with elitism)
    print("\nTesting mutation rates")
    for mutation_std in mutation_stds:
        generations_list = []
        for trial in range(n_trials):
            optimizer = EvolutionaryOptimizer(
                selection_size=10,  # fixed selection size
                use_elitism=True,
                mutation_std=mutation_std,
            )
            _, history = optimizer.optimize(selection_method=selection_method)
            generations_list.append(len(history))
        results["mutation_study"][mutation_std] = np.mean(generations_list)

    return results


def compare_methods():
    methods = ["truncation", "roulette", "tournament"]
    selection_sizes = [2, 5, 10, 20]
    mutation_stds = [0.01, 0.05, 0.1, 0.2]

    all_results = {}
    method_times = {}

    # run evolutionary methods
    for method in methods:
        print(f"\nTesting {method} selection")
        start_time = time.time()
        results = run_experiment(method, selection_sizes, mutation_stds)
        method_times[method] = time.time() - start_time
        all_results[method] = results

    # run Twiddle for comparison
    print("\nRunning Twiddle")
    start_time = time.time()
    # run your Twiddle implementation here
    twiddle_time = time.time() - start_time
    method_times["twiddle"] = twiddle_time

    # plot results
    # 1. Selection size comparison
    plt.figure(figsize=(12, 6))
    colors = ["b", "g", "r"]
    for i, method in enumerate(methods):
        results = all_results[method]
        plt.plot(
            selection_sizes,
            [results["with_elitism"][s] for s in selection_sizes],
            f"{colors[i]}-",
            label=f"{method} with elitism",
        )
        plt.plot(
            selection_sizes,
            [results["without_elitism"][s] for s in selection_sizes],
            f"{colors[i]}--",
            label=f"{method} without elitism",
        )

    plt.xlabel("Selection Size")
    plt.ylabel("Average Generations to Converge")
    plt.title("Effect of Selection Size on Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig("selection_size_comparison.png")
    plt.show()

    # 2. Mutation rate comparison
    plt.figure(figsize=(12, 6))
    for i, method in enumerate(methods):
        results = all_results[method]
        plt.plot(
            mutation_stds,
            [results["mutation_study"][m] for m in mutation_stds],
            f"{colors[i]}-o",
            label=method,
        )

    plt.xlabel("Mutation Standard Deviation")
    plt.ylabel("Average Generations to Converge")
    plt.title("Effect of Mutation Rate on Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig("mutation_rate_comparison.png")
    plt.show()

    # 3. Wall time comparison
    plt.figure(figsize=(8, 6))
    methods_with_twiddle = methods + ["twiddle"]
    plt.bar(methods_with_twiddle, [method_times[m] for m in methods_with_twiddle])
    plt.ylabel("Wall Time (seconds)")
    plt.title("Algorithm Wall Time Comparison")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig("wall_time_comparison.png")
    plt.show()


if __name__ == "__main__":
    compare_methods()
