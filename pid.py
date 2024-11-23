import random

class PIDcontroller:
    def __init__(self, params=[0.02, 0.0001, 0.05]):
        self.p_p = params[0]  # proportional gain
        self.p_i = params[1]  # integral gain
        self.p_d = params[2]  # derivative gain
        self.prev_cte = 0
        self.integral = 0
        self.integral_max = 100

    def reset(self):
        self.prev_cte = 0
        self.integral = 0

    def process(self, CTE):
        # derivative of CTE
        cte_derivative = CTE - self.prev_cte
        self.prev_cte = CTE

        # update integral term with anti-windup
        self.integral += CTE
        self.integral = max(min(self.integral, self.integral_max), -self.integral_max)

        steering = (
            -self.p_p * CTE - self.p_d * cte_derivative - self.p_i * self.integral
        )
        return steering


class Twiddle:
    def __init__(self):
        self.params = [0.02, 0.0001, 0.05]  # initial [p, i, d]
        self.dp = [0.002, 0.00001, 0.005]  # init parameter deltas
        self.best_error = float("inf")

    def run_iteration(self, evaluate_func):
        for i in range(len(self.params)):
            # try increasing param
            self.params[i] += self.dp[i]
            error = evaluate_func(self.params)

            if error < self.best_error:
                self.best_error = error
                self.dp[i] *= 1.1
            else:
                # try decreasing param
                self.params[i] -= 2 * self.dp[i]
                error = evaluate_func(self.params)

                if error < self.best_error:
                    self.best_error = error
                    self.dp[i] *= 1.1
                else:
                    # revert and decrease step size
                    self.params[i] += self.dp[i]
                    self.dp[i] *= 0.9

        return self.params, self.best_error

class EvolutionaryAlgorithm:
    def __init__(self, population_size = 100, mutation_rate=0.1, crossover_rate=0.5):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = [
            [random.uniform(0.01, 0.1), random.uniform(0.00001, 0.001), random.uniform(0.01, 0.1)]
            for _ in range(population_size)
        ]
        self.best_individual = None
        self.best_error = float("inf")

    def mutate(self, params):
        return [
            param + random.uniform(-param * self.mutation_rate, param * self.mutation_rate)
            for param in params
        ]
        
    def crossover(self, parent1, parent2):
        child = [
            parent1[i] if random.random() < self.crossover_rate else parent2[i]
            for i in range(len(parent1))
        ]
        return child
        
    def select_parents(self, population, fitnesses):
        total_fitness = sum(fitnesses)
        probs = [1.0 - (f / total_fitness) for f in fitnesses]
        return random.choices(population, probs, k=2)
        
    def run_iteration(self, evaluate_func):
        fitnesses = [evaluate_func(ind) for ind in self.population]

        for i, error in enumerate(fitnesses):
            if error < self.best_error:
                self.best_error = error
                self.best_individual = self.population[i]

        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = self.select_parents(self.population, fitnesses)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population
        return self.best_individual, self.best_error