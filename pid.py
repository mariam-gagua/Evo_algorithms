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
