import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from main import evaluate_controller


def analyze_error_landscape():
    # default params
    default_p = 0.02
    default_i = 0.0001
    default_d = 0.05

    # param ranges
    p_range = np.linspace(0, 0.1, 30)
    i_range = np.linspace(0, 0.001, 30)
    d_range = np.linspace(0, 0.2, 30)

    # meshgrids for each combination
    P, I = np.meshgrid(p_range, i_range)
    P, D = np.meshgrid(p_range, d_range)
    I, D = np.meshgrid(i_range, d_range)

    # init error surfaces
    error_PI = np.zeros_like(P)
    error_PD = np.zeros_like(P)
    error_ID = np.zeros_like(I)

    # error surfaces
    print("Calculating error surfaces... This may take a while...")
    for i in range(len(p_range)):
        for j in range(len(i_range)):
            # P-I surface (D fixed)
            error_PI[j, i] = evaluate_controller([P[j, i], I[j, i], default_d])

            # P-D surface (I fixed)
            error_PD[j, i] = evaluate_controller([P[j, i], default_i, D[j, i]])

            # I-D surface (P fixed)
            error_ID[j, i] = evaluate_controller([default_p, I[j, i], D[j, i]])

    fig = plt.figure(figsize=(20, 6))

    # P-I surface plot
    ax1 = fig.add_subplot(131, projection="3d")
    surf1 = ax1.plot_surface(P, I, error_PI, cmap="viridis")
    ax1.set_xlabel("P Parameter")
    ax1.set_ylabel("I Parameter")
    ax1.set_zlabel("Error")
    ax1.set_title("Error Landscape: P-I (D fixed)")

    # P-D surface plot
    ax2 = fig.add_subplot(132, projection="3d")
    surf2 = ax2.plot_surface(P, D, error_PD, cmap="viridis")
    ax2.set_xlabel("P Parameter")
    ax2.set_ylabel("D Parameter")
    ax2.set_zlabel("Error")
    ax2.set_title("Error Landscape: P-D (I fixed)")

    # I-D surface plot
    ax3 = fig.add_subplot(133, projection="3d")
    surf3 = ax3.plot_surface(I, D, error_ID, cmap="viridis")
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


if __name__ == "__main__":
    analyze_error_landscape()
