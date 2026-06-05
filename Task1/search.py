from problem import *
import math
import numpy as np
import random

INIT_TEMP   = 1000
ALPHA       = 0.999
TIME        = 5000
FACTOR      = 20.0
EPSILON     = 0.1

class Schedule:
    """Manages the cooling schedule (temperature decay) for the algorithm."""

    def __init__(self, T0=INIT_TEMP):
        """Initializes the schedule with a starting temperature."""
        self.T0 = T0 
        self.T = T0 

    def sd1(self, alpha=ALPHA):
        """Standard geometric cooling: strictly decays temperature by a constant factor."""
        self.T = self.T * alpha 
        return self.T 
    
    def sd2(self, delta_z, t_max=TIME, t0=INIT_TEMP, epsilon=EPSILON, alpha=ALPHA, k=FACTOR):
        """Adaptive cooling: decays normally, but injects heat (reheats) when stuck in flat regions."""
        base_t = self.sd1(alpha)
        delta_z = abs(delta_z)
        additional = k / (delta_z + epsilon)
        self.T = base_t + additional
        return self.T
    
    def accept(self, delta_z):
        """Evaluates whether to accept a move based on energy change and current temperature."""
        if delta_z > 0:
            return True 
        if self.T < 1e-8:
            return False 
        return random.random() < math.exp(max(delta_z / self.T, -100))
    
    def compute_T(self, delta_z, sched_name):
        """Routes the temperature calculation to the selected schedule function."""
        if sched_name == "sd1":
            return self.sd1()
        else:
            return self.sd2(delta_z)
        
    def reset_T(self):
        """Resets the current temperature back to the initial value."""
        self.T = self.T0


class Stimulated_Annealing_Search:
    """Executes the Simulated Annealing search algorithm on a provided problem."""
    
    def __init__(self, prob: Problem, sched_func: str):
        """Initializes the searcher with a problem space and schedule type ('sd1' or 'sd2')."""
        self.initial_state = prob.get_initial_state()
        self.this_problem = prob
        self.actual_step = None
        self.sched_func = str(sched_func)
    
    def search(self, sched: Schedule, loop):
        """
        Runs the main optimization loop.
        
        Returns:
            tuple: A tuple containing:
                - path (list): The sequence of accepted states.
                - best_state (State): The absolute best state discovered during the run.
        """
        current = self.this_problem.get_initial_state()
        best_state = current
        path = [current]
        step = 0

        for t in range(loop):
            neighbors = self.this_problem.get_neighbors(current)

            if not neighbors or sched.T < 1e-8:
                break
            
            candidate = random.choice(neighbors)
            delta_E = np.float64(candidate.get_value()) - np.float64(current.get_value())
            sched.T = sched.compute_T(delta_E, self.sched_func)

            if best_state.get_value() < candidate.get_value():
                best_state = candidate

            if sched.accept(delta_E):
                current = candidate
                path.append(current)
                step += 1

        self.actual_step = step
        return path, best_state
