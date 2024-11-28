import numpy as np
import matplotlib.pyplot as plt
from evolutionary_optimization import EvolutionaryOptimizer
from main import evaluate_controller
import time
from scipy import stats


def run_experiment(
    selection_method,
    selection_sizes=[2, 5, 10, 20],
    mutation_stds=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    n_trials=10,
):  # Increased trials
    results = {
        "with_elitism": {size: {"mean": 0, "std": 0} for size in selection_sizes},
        "without_elitism": {size: {"mean": 0, "std": 0} for size in selection_sizes},
        "mutation_study": {mut: {"mean": 0, "std": 0} for mut in mutation_stds},
    }

    total_experiments = (
        len(selection_sizes) * 2 * n_trials + len(mutation_stds) * n_trials
    )
    current_experiment = 0

    # Test selection sizes with and without elitism
    for selection_size in selection_sizes:
        print(f"\nTesting {selection_method} selection with size {selection_size}")
        for use_elitism in [True, False]:
            generations_list = []
            for trial in range(n_trials):
                current_experiment += 1
                print(f"Progress: {current_experiment}/{total_experiments}")

                optimizer = EvolutionaryOptimizer(
                    selection_size=selection_size,
                    use_elitism=use_elitism,
                    mutation_std=0.1,
                )
                _, history = optimizer.optimize(
                    selection_method=selection_method, max_generations=50
                )
                generations_list.append(len(history))

            key = "with_elitism" if use_elitism else "without_elitism"
            results[key][selection_size] = {
                "mean": np.mean(generations_list),
                "std": np.std(generations_list),
            }

    # Test mutation rates (with elitism only)
    print(f"\nTesting {selection_method} mutation rates")
    for mutation_std in mutation_stds:
        generations_list = []
        for trial in range(n_trials):
            current_experiment += 1
            print(f"Progress: {current_experiment}/{total_experiments}")

            optimizer = EvolutionaryOptimizer(
                selection_size=10, use_elitism=True, mutation_std=mutation_std
            )
            _, history = optimizer.optimize(
                selection_method=selection_method, max_generations=50
            )
            generations_list.append(len(history))
        results["mutation_study"][mutation_std] = {
            "mean": np.mean(generations_list),
            "std": np.std(generations_list),
        }

    return results


def run_twiddle(n_trials=10):
    times = []
    errors = []
    for trial in range(n_trials):
        print(f"Running Twiddle trial {trial + 1}/{n_trials}")
        start_time = time.time()
        # Run your Twiddle implementation here
        from main import evaluate_controller  # Replace with your Twiddle implementation

        end_time = time.time()
        times.append(end_time - start_time)
        # Store final error if available
    return np.mean(times), np.std(times)


def compare_methods():
    methods = ["truncation", "roulette", "tournament"]
    all_results = {}
    method_times = {}
    method_time_stds = {}

    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing {method} selection")
        print(f"{'='*50}")

        # Run multiple trials for timing
        times = []
        for _ in range(5):  # 5 timing trials
            start_time = time.time()
            results = run_experiment(method)
            times.append(time.time() - start_time)

        method_times[method] = np.mean(times)
        method_time_stds[method] = np.std(times)
        all_results[method] = results

        # Save intermediate results
        plot_results(
            all_results, method_times, method_time_stds, f"results_{method}.png"
        )

    # Add Twiddle timing
    method_times["twiddle"], method_time_stds["twiddle"] = run_twiddle()

    return all_results, method_times, method_time_stds


def plot_results(all_results, method_times, method_time_stds, filename="results.png"):
    fig = plt.figure(figsize=(20, 6))
    gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1])

    # Plot selection size comparison
    ax1 = fig.add_subplot(gs[0])
    selection_sizes = [2, 5, 10, 20]
    colors = ["b", "g", "r"]
    for i, (method, results) in enumerate(all_results.items()):
        means_with = [results["with_elitism"][s]["mean"] for s in selection_sizes]
        stds_with = [results["with_elitism"][s]["std"] for s in selection_sizes]
        means_without = [results["without_elitism"][s]["mean"] for s in selection_sizes]
        stds_without = [results["without_elitism"][s]["std"] for s in selection_sizes]

        ax1.errorbar(
            selection_sizes,
            means_with,
            yerr=stds_with,
            fmt=f"{colors[i]}-",
            label=f"{method} with elitism",
        )
        ax1.errorbar(
            selection_sizes,
            means_without,
            yerr=stds_without,
            fmt=f"{colors[i]}--",
            label=f"{method} without elitism",
        )

    ax1.set_xlabel("Selection Size")
    ax1.set_ylabel("Average Generations to Converge")
    ax1.set_title("Effect of Selection Size")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True)

    # Plot mutation rate comparison
    ax2 = fig.add_subplot(gs[1])
    mutation_stds = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
    for i, (method, results) in enumerate(all_results.items()):
        means = [results["mutation_study"][m]["mean"] for m in mutation_stds]
        stds = [results["mutation_study"][m]["std"] for m in mutation_stds]
        ax2.errorbar(
            mutation_stds, means, yerr=stds, fmt=f"{colors[i]}-o", label=method
        )

    ax2.set_xlabel("Mutation Standard Deviation")
    ax2.set_ylabel("Average Generations to Converge")
    ax2.set_title("Effect of Mutation Rate")
    ax2.legend()
    ax2.grid(True)

    # Plot wall time comparison with error bars
    ax3 = fig.add_subplot(gs[2])
    methods = list(method_times.keys())
    times = [method_times[m] for m in methods]
    time_stds = [method_time_stds[m] for m in methods]

    bars = ax3.bar(methods, times, yerr=time_stds, capsize=5)
    ax3.set_ylabel("Wall Time (seconds)")
    ax3.set_title("Algorithm Wall Time")
    ax3.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}s",
            ha="center",
            va="bottom",
        )

    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    all_results, method_times, method_time_stds = compare_methods()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("==================")
    for method in all_results.keys():
        print(f"\n{method.upper()} Selection:")
        print(
            f"Average wall time: {method_times[method]:.2f}s ± {method_time_stds[method]:.2f}s"
        )

        # Best configuration
        best_size = min(
            all_results[method]["with_elitism"].items(), key=lambda x: x[1]["mean"]
        )[0]
        best_mutation = min(
            all_results[method]["mutation_study"].items(), key=lambda x: x[1]["mean"]
        )[0]

        print(f"Best selection size: {best_size}")
        print(f"Best mutation rate: {best_mutation:.3f}")
        print(
            f"Best generations to converge: "
            f"{all_results[method]['with_elitism'][best_size]['mean']:.1f} ± "
            f"{all_results[method]['with_elitism'][best_size]['std']:.1f}"
        )
