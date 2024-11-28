import numpy as np
import matplotlib.pyplot as plt
from main import evaluate_controller


def analyze_error_landscape():
    # Parameter ranges
    default_p, default_i, default_d = 0.14, 0.003, 6.1
    p_range = np.linspace(0.06, 0.20, 20)
    i_range = np.linspace(0.001, 0.005, 20)
    d_range = np.linspace(2.0, 7.0, 20)

    # Create meshgrids
    P, I = np.meshgrid(p_range, i_range)
    P_d, D = np.meshgrid(p_range, d_range)
    I_d, D_i = np.meshgrid(i_range, d_range)

    # Calculate error surfaces
    error_PI = np.zeros_like(P)
    error_PD = np.zeros_like(P_d)
    error_ID = np.zeros_like(I_d)

    print("Calculating error surfaces...")

    print("Calculating P-I surface...")
    # P-I surface (D fixed)
    for i in range(len(p_range)):
        for j in range(len(i_range)):
            try:
                error = evaluate_controller([P[j, i], I[j, i], default_d])
                error_PI[j, i] = error
                if i % 5 == 0 and j % 5 == 0:
                    print(f"P={P[j,i]:.4f}, I={I[j,i]:.4f}, Error={error:.4f}")
            except Exception as e:
                print(f"Error at P={P[j,i]}, I={I[j,i]}: {str(e)}")
                error_PI[j, i] = float("inf")

    # P-D surface (I fixed)
    for i in range(len(p_range)):
        for j in range(len(d_range)):
            error_PD[j, i] = evaluate_controller([P_d[j, i], default_i, D[j, i]])

    # I-D surface (P fixed)
    for i in range(len(i_range)):
        for j in range(len(d_range)):
            error_ID[j, i] = evaluate_controller([default_p, I_d[j, i], D_i[j, i]])

    # Plot surfaces
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(P, I, error_PI, cmap="viridis")
    ax1.set_xlabel("P Parameter")
    ax1.set_ylabel("I Parameter")
    ax1.set_zlabel("Error")
    ax1.set_title("P-I Error Landscape")

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.plot_surface(P_d, D, error_PD, cmap="viridis")
    ax2.set_xlabel("P Parameter")
    ax2.set_ylabel("D Parameter")
    ax2.set_zlabel("Error")
    ax2.set_title("P-D Error Landscape")

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot_surface(I_d, D_i, error_ID, cmap="viridis")
    ax3.set_xlabel("I Parameter")
    ax3.set_ylabel("D Parameter")
    ax3.set_zlabel("Error")
    ax3.set_title("I-D Error Landscape")

    plt.tight_layout()
    plt.savefig("error_landscape.png")
    plt.close()

    # Print some statistics about the surfaces
    print("\nError surface statistics:")
    print(f"P-I surface - Min: {np.min(error_PI)}, Max: {np.max(error_PI)}")
    print(f"P-D surface - Min: {np.min(error_PD)}, Max: {np.max(error_PD)}")
    print(f"I-D surface - Min: {np.min(error_ID)}, Max: {np.max(error_ID)}")


if __name__ == "__main__":
    analyze_error_landscape()
