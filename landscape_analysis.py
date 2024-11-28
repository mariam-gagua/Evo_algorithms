import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from main import evaluate_controller
from scipy.ndimage import gaussian_filter


def analyze_error_landscape():
    # Use the best parameters found by Twiddle as default values
    default_p = 0.14
    default_i = 0.003
    default_d = 6.1

    # Modify ranges to be more reasonable and centered around defaults
    p_range = np.linspace(0.06, 0.20, 20)  # Reduced number of points, adjusted range
    i_range = np.linspace(0.001, 0.005, 20)  # Reduced number of points
    d_range = np.linspace(2.0, 7.0, 20)  # Reduced number of points

    # meshgrids for each combination
    P, I = np.meshgrid(p_range, i_range)
    P_d, D = np.meshgrid(p_range, d_range)
    I_d, D_i = np.meshgrid(i_range, d_range)

    # init error surfaces with correct shapes
    error_PI = np.zeros_like(P, dtype=float)
    error_PD = np.zeros_like(P_d, dtype=float)
    error_ID = np.zeros_like(I_d, dtype=float)

    # error surfaces
    print("Calculating error surfaces... This may take a while...")

    # Calculate P-I surface
    for i in range(len(p_range)):
        for j in range(len(i_range)):
            try:
                params = [P[j, i], I[j, i], default_d]
                error = evaluate_controller(params)
                error_PI[j, i] = min(error, 50000)  # Lower cap on maximum error

                # Print progress more frequently
                if i % 5 == 0 and j % 5 == 0:
                    print(
                        f"P-I Surface: P={P[j,i]:.4f}, I={I[j,i]:.4f}, D={default_d:.4f}, Error={error_PI[j,i]:.4f}"
                    )
            except Exception as e:
                print(f"Error at P={P[j,i]}, I={I[j,i]}: {str(e)}")
                error_PI[j, i] = 50000

    # Calculate P-D surface
    for i in range(len(p_range)):
        for j in range(len(d_range)):
            try:
                params = [P_d[j, i], default_i, D[j, i]]
                error = evaluate_controller(params)
                error_PD[j, i] = min(error, 50000)
            except Exception as e:
                print(f"Error at P={P_d[j,i]}, D={D[j,i]}: {str(e)}")
                error_PD[j, i] = 50000

    # Calculate I-D surface
    for i in range(len(i_range)):
        for j in range(len(d_range)):
            try:
                params = [default_p, I_d[j, i], D_i[j, i]]
                error = evaluate_controller(params)
                error_ID[j, i] = min(error, 50000)
            except Exception as e:
                print(f"Error at I={I_d[j,i]}, D={D_i[j,i]}: {str(e)}")
                error_ID[j, i] = 50000

    # Modify smoothing to be less aggressive
    error_PI = gaussian_filter(error_PI, sigma=0.5)
    error_PD = gaussian_filter(error_PD, sigma=0.5)
    error_ID = gaussian_filter(error_ID, sigma=0.5)

    # Modify normalization to handle extreme values better
    def normalize_surface(surface):
        valid_mask = surface < 50000
        if not np.any(valid_mask):
            return surface
        p1 = np.percentile(surface[valid_mask], 1)
        p99 = np.percentile(surface[valid_mask], 99)
        normalized = np.clip(surface, p1, p99)
        return normalized

    error_PI = normalize_surface(error_PI)
    error_PD = normalize_surface(error_PD)
    error_ID = normalize_surface(error_ID)

    # Create figure with smaller size
    fig = plt.figure(figsize=(15, 5))

    # P-I surface plot
    ax1 = fig.add_subplot(131, projection="3d")
    surf1 = ax1.plot_surface(P, I, error_PI, cmap="viridis")
    ax1.set_xlabel("P Parameter")
    ax1.set_ylabel("I Parameter")
    ax1.set_zlabel("Error")
    ax1.set_title("Error Landscape: P-I (D fixed)")

    # P-D surface plot
    ax2 = fig.add_subplot(132, projection="3d")
    surf2 = ax2.plot_surface(P_d, D, error_PD, cmap="viridis")
    ax2.set_xlabel("P Parameter")
    ax2.set_ylabel("D Parameter")
    ax2.set_zlabel("Error")
    ax2.set_title("Error Landscape: P-D (I fixed)")

    # I-D surface plot
    ax3 = fig.add_subplot(133, projection="3d")
    surf3 = ax3.plot_surface(I_d, D_i, error_ID, cmap="viridis")
    ax3.set_xlabel("I Parameter")
    ax3.set_ylabel("D Parameter")
    ax3.set_zlabel("Error")
    ax3.set_title("Error Landscape: I-D (P fixed)")

    plt.tight_layout()
    plt.show()

    # analyze local minima
    def find_local_minima(error_surface):
        local_minima = []
        rows, cols = error_surface.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                current = error_surface[i, j]
                neighbors = [
                    error_surface[i - 1, j],
                    error_surface[i + 1, j],
                    error_surface[i, j - 1],
                    error_surface[i, j + 1],
                ]
                if all(current < neighbor for neighbor in neighbors):
                    local_minima.append((i, j))
        return local_minima

    pi_minima = find_local_minima(error_PI)
    pd_minima = find_local_minima(error_PD)
    id_minima = find_local_minima(error_ID)

    print("\nAnalysis Results:")
    print(f"Number of local minima found:")
    print(f"P-I surface: {len(pi_minima)}")
    print(f"P-D surface: {len(pd_minima)}")
    print(f"I-D surface: {len(id_minima)}")

    # Add contour plots below each 3D surface for better visualization of local minima
    fig = plt.figure(figsize=(20, 12))  # Increased height for additional plots

    # P-I surface and contour
    ax1 = fig.add_subplot(231, projection="3d")
    ax1_contour = fig.add_subplot(234)
    surf1 = ax1.plot_surface(P, I, error_PI, cmap="viridis")
    cont1 = ax1_contour.contour(P, I, error_PI, levels=20)
    ax1_contour.clabel(cont1, inline=True, fontsize=8)

    # Similar updates for P-D and I-D plots...

    # Add analysis of convexity
    def check_convexity(error_surface):
        # Simple check - if all local minima are within small range
        # of global minimum, surface is approximately convex
        global_min = np.min(error_surface)
        local_minima_vals = [
            error_surface[i, j] for i, j in find_local_minima(error_surface)
        ]
        if not local_minima_vals:
            return "No local minima found"
        max_deviation = max([abs(v - global_min) for v in local_minima_vals])
        return max_deviation < 0.1 * global_min

    print("\nConvexity Analysis:")
    print(f"P-I surface convex: {check_convexity(error_PI)}")
    print(f"P-D surface convex: {check_convexity(error_PD)}")
    print(f"I-D surface convex: {check_convexity(error_ID)}")


def plot_convergence_comparison():
    """Compare convergence speed of different selection methods"""
    methods = ["truncation", "roulette", "tournament"]
    generations_mean = []
    generations_std = []

    n_trials = 10
    for method in methods:
        gens = []
        for i in range(n_trials):
            _, generations = optimize_evo(selection_method=method)
            gens.append(generations)
        generations_mean.append(np.mean(gens))
        generations_std.append(np.std(gens))

    plt.figure(figsize=(10, 6))
    plt.bar(methods, generations_mean, yerr=generations_std, capsize=5)
    plt.xlabel("Selection Method")
    plt.ylabel("Generations to Converge")
    plt.title("Selection Method Convergence Speed Comparison")
    plt.grid(True)
    plt.savefig("convergence_comparison.png")
    plt.close()


if __name__ == "__main__":
    analyze_error_landscape()
