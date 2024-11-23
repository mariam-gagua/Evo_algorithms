import math
import pygame
import pid
from utils import scale_image, blit_rotate_center, blit_text_center
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import time 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


pygame.init()
pygame.font.init()


clock = pygame.time.Clock()
debug = False

wind = False
steer_bias = True

FrameHeight = 400
FrameWidth = 1200

pygame.display.set_caption("PID controller simulation")
screen = pygame.display.set_mode((FrameWidth, FrameHeight))

bg = pygame.image.load("background_small.png").convert()
RED_CAR = scale_image(pygame.image.load("imgs/red-car_small.png"), 1.0)

MAIN_FONT = pygame.font.SysFont("courier", 35)


def draw(win, player_car, scroll):

    i = 0
    while i < tiles:
        screen.blit(bg, (bg.get_width() * i + scroll, 0))
        i += 1

    # RESET THE SCROLL FRAME
    if abs(scroll) > bg.get_width():
        scroll = 0

    if debug:
        level_text = MAIN_FONT.render(f"CTE {player_car.y - 266}", 1, (255, 255, 255))
        win.blit(level_text, (10, FrameHeight - level_text.get_height() - 70))

        steer_text = MAIN_FONT.render(
            f"Steering angle: {player_car.steering_angle}", 1, (255, 255, 255)
        )
        win.blit(steer_text, (10, FrameHeight - steer_text.get_height() - 40))

        vel_text = MAIN_FONT.render(
            f"Vel: {round(player_car.vel, 1)} px/s", 1, (255, 255, 255)
        )
        win.blit(vel_text, (10, FrameHeight - vel_text.get_height() - 10))
        print(player_car.x)

    player_car.draw(win)
    pygame.display.update()

    return scroll


def move_player(player_car):

    keys = pygame.key.get_pressed()
    moved = False

    current_CTE = player_car.y - 266
    print(f"CTE = {current_CTE}")

    player_car.steering_angle = controller.process(current_CTE)

    if steer_bias:
        player_car.steering_angle += 0.3

    player_car.rotate()

    if debug:
        if keys[pygame.K_w]:
            moved = True
            player_car.move_forward()
        if keys[pygame.K_s]:
            moved = True
            player_car.move_backward()
        if not moved:
            player_car.reduce_speed()

    if not debug:
        player_car.move_forward()


class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.max_steering_angle = 4.0
        self.steering_angle = 0.0
        self.angle = 220
        self.x, self.y = self.START_POS
        self.prev_x, self.prev_y = self.START_POS
        self.acceleration = 0.1

    def rotate(self):
        if self.steering_angle > self.max_steering_angle:
            self.steering_angle = self.max_steering_angle
        if self.steering_angle < -self.max_steering_angle:
            self.steering_angle = -self.max_steering_angle

        # test for velocity-related steering speed, uncomment for original
        self.angle -= (self.vel / self.max_vel) * self.steering_angle

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel / 2)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.prev_x = self.x
        self.prev_y = self.y
        self.y -= vertical
        self.x -= horizontal

        if wind:
            self.y -= 0.2

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0


class PlayerCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (45, 200)

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

    def bounce(self):
        self.vel = -self.vel
        self.move()


player_car = PlayerCar(1, 4)
controller = pid.PIDcontroller()

scroll = 0

tiles = math.ceil(FrameWidth / bg.get_width()) + 1


def evaluate_controller(params, max_steps=1000):
    # reset simulation
    player_car = PlayerCar(1, 4)
    controller = pid.PIDcontroller(params)
    total_error = 0

    for step in range(max_steps):
        current_CTE = player_car.y - 266
        player_car.steering_angle = controller.process(current_CTE)

        if steer_bias:
            player_car.steering_angle += 0.3

        player_car.rotate()
        player_car.move_forward()

        # accumulate abs error
        total_error += abs(current_CTE)

        # off screen case - terminate
        if (
            player_car.x > 1200
            or player_car.x < 0
            or player_car.y < 0
            or player_car.y > 400
        ):
            total_error += (max_steps - step) * 100  # penalty
            break

    return total_error

0.02, 0.0001, 0.05


# Define ranges for parameters
P_range = np.linspace(0, 0.01, 50)
I_range = np.linspace(0, 0.0001, 50)
D_range = np.linspace(0, 0.01, 50)

# Fixed parameter values
fixed_I = 0.0001
fixed_D = 0.05
fixed_P = 0.02

# Create grids for two parameters while keeping the third fixed
P, I = np.meshgrid(P_range, I_range)
D, I = np.meshgrid(D_range, I_range)
P, D = np.meshgrid(P_range, D_range)

# Compute error for each grid combination (replace with evaluate_controller)
error_PI = np.array([[evaluate_controller([p, i, fixed_D]) for p, i in zip(row_p, row_i)] for row_p, row_i in zip(P, I)])
error_PD = np.array([[evaluate_controller([p, fixed_I, d]) for p, d in zip(row_p, row_d)] for row_p, row_d in zip(P, D)])
error_ID = np.array([[evaluate_controller([fixed_P, i, d]) for i, d in zip(row_i, row_d)] for row_i, row_d in zip(I, D)])

# Plot surfaces
fig = plt.figure(figsize=(15, 5))

# P-I
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(P, I, error_PI, cmap='viridis')
ax1.set_title('Error Surface for P and I')
ax1.set_xlabel('P')
ax1.set_ylabel('I')
ax1.set_zlabel('Error')

# P-D
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(P, D, error_PD, cmap='viridis')
ax2.set_title('Error Surface for P and D')
ax2.set_xlabel('P')
ax2.set_ylabel('D')
ax2.set_zlabel('Error')

# I-D
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(I, D, error_ID, cmap='viridis')
ax3.set_title('Error Surface for I and D')
ax3.set_xlabel('I')
ax3.set_ylabel('D')
ax3.set_zlabel('Error')

plt.show()

# # twiddle 
# print("Optimizing started...")
# twiddle = pid.Twiddle()
# best_params = None
# best_error = float("inf")
# num_iterations = 100

# for i in range(num_iterations):
#     params, error = twiddle.run_iteration(evaluate_controller)
#     if error < best_error:
#         best_params = params.copy()
#         best_error = error
#     print(f"Iteration {i}: Parameters: {params}, Error: {error}")

# print(f"\nBest parameters found: {best_params}")

# # run the car with best params
# print("Running simulation with best parameters...")
# player_car = PlayerCar(1, 4)
# controller = pid.PIDcontroller(best_params)
# scroll = 0

# # visualization loop
# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     clock.tick(60)
#     scroll = draw(screen, player_car, scroll)
#     move_player(player_car)

#     if (
#         player_car.x > 1200
#         or player_car.x < 0
#         or player_car.y < 0
#         or player_car.y > 400
#     ):
#         player_car = PlayerCar(1, 4)
#         controller.reset()

# pygame.quit()

# evolutionary algorithm
print("ptimizing started...")
evolutionary_algo = pid.EvolutionaryAlgorithm(population_size=100, mutation_rate=0.1, crossover_rate=0.5)
best_params = None
best_error = float("inf")
num_iterations = 100

for i in range(num_iterations):
    params, error = evolutionary_algo.run_iteration(evaluate_controller)
    if error < best_error:
        best_params = params.copy()
        best_error = error
    print(f"Generation {i}: Parameters: {params}, Error: {error}")

print(f"\nBest parameters found: {best_params}")

# run the car with best params
print("Running simulation with best parameters...")
player_car = PlayerCar(1, 4)
controller = pid.PIDcontroller(best_params)
scroll = 0

# visualization loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    clock.tick(60)
    scroll = draw(screen, player_car, scroll)
    move_player(player_car)

    if (
        player_car.x > 1200
        or player_car.x < 0
        or player_car.y < 0
        or player_car.y > 400
    ):
        player_car = PlayerCar(1, 4)
        controller.reset()

pygame.quit()
