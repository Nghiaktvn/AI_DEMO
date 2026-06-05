from problem import *
from search import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

IMAGE_PATH          = "monalisa.jpg"
SCHEDULE_FUNCTION_1 = "sd1"
SCHEDULE_FUNCTION_2 = "sd2"
EXPERIMENT          = 100
LOOP                = 5000

class Tester:
    """Manages the execution of Simulated Annealing experiments and generates statistical reports and visualizations."""

    def __init__(self, experiment_time, loop_time):
        """Initializes the tester with the total number of runs and maximum iterations per run."""
        self.experiment_time = experiment_time 
        self.loop_time = loop_time 

    def multiple_run(self, problem, schedule_func):
        """
        Executes the algorithm multiple times to gather statistical data.
        
        Returns:
            list: A list of dictionaries containing the best 'z' and total 'steps' for each run.
        """
        _, _, Z = problem.load_state_space()
        moved = []

        for _ in range(self.experiment_time):
            scheduler = Schedule()
            searcher = Stimulated_Annealing_Search(problem, schedule_func)
            path, _ = searcher.search(scheduler, LOOP)

            best_z = max(s.get_attribute()[2] for s in path) if path else 0
            moved.append({
                'z':    best_z,
                'steps': searcher.actual_step,
            })

        return moved
    
    def single_run(self, problem, schedule_func):
        """
        Executes a single run of the algorithm to track the step-by-step convergence history.
        
        Returns:
            list: The history of the best 'z' values found at each step.
        """
        scheduler = Schedule()
    
        searcher = Stimulated_Annealing_Search(problem, schedule_func)
        path, best_state = searcher.search(scheduler, LOOP)
    
        history_z = []
        best_so_far = -np.inf 
        for state in path:
            z = state.get_attribute()[2]
            if z > best_so_far:
                best_so_far = z 
            history_z.append(best_so_far)

        problem.visualize_space(path, best_state, schedule_func)

        return history_z 
    
    def generate_summary_table(self, results_sd1, results_sd2):
        """Computes and prints a pandas DataFrame summarizing the statistical performance of both schedules."""

        def stats(results):
            z     = [r['z']     for r in results]
            steps = [r['steps'] for r in results]
            return {
                'Max Z':        np.max(z),
                'Mean Z':       np.mean(z),
                'Std Z':        np.std(z),
                'Mean Steps':   np.mean(steps),
                'Max Steps':    np.max(steps),
            }

        df = pd.DataFrame(
            [stats(results_sd1), stats(results_sd2)],
            index=['sd1 (exponential)', 'sd2 (slope-adaptive)']
        )

        print("\n" + "=" * 60)
        print("   SUMMARY TABLE: SD1 vs SD2")
        print("=" * 60)
        print(df.round(2).to_string())
        print("=" * 60)
        return df

    def generate_figures(self, results_sd1, results_sd2,
                         history_sd1_sample, history_sd2_sample):
        """
        Generates and saves a 3-panel comparative chart (Convergence Line, Best Z Boxplot, Steps Boxplot).
        
        Args:
            results_sd1: Output from multiple_run() for sd1.
            results_sd2: Output from multiple_run() for sd2.
            history_sd1_sample: Output from single_run() for sd1.
            history_sd2_sample: Output from single_run() for sd2.
        """
        z_sd1    = [r['z']     for r in results_sd1]
        z_sd2    = [r['z']     for r in results_sd2]
        step_sd1 = [r['steps'] for r in results_sd1]
        step_sd2 = [r['steps'] for r in results_sd2]

        n = self.experiment_time
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # ── panel 1: convergence line chart ───────
        ax = axes[0]
        ax.plot(history_sd1_sample, label='sd1', color='steelblue', linewidth=1.5)
        ax.plot(history_sd2_sample, label='sd2', color='tomato',    linewidth=1.5, alpha=0.85)
        ax.set_title('Convergence (one representative run)')
        ax.set_xlabel('Time step t')
        ax.set_ylabel('Best z found so far')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)

        # ── panel 2: best z boxplot ────────────────
        ax = axes[1]
        bp = ax.boxplot([z_sd1, z_sd2], tick_labels=['sd1', 'sd2'],
                        patch_artist=True, notch=False)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightsalmon']):
            patch.set_facecolor(color)
        ax.set_title(f'Best Z Distribution  (n={n} runs)')
        ax.set_ylabel('Best Z value found')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

        # ── panel 3: steps boxplot ─────────────────
        ax = axes[2]
        bp = ax.boxplot([step_sd1, step_sd2], tick_labels=['sd1', 'sd2'],
                        patch_artist=True, notch=False)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightsalmon']):
            patch.set_facecolor(color)
        ax.set_title(f'Steps Distribution  (n={n} runs)')
        ax.set_ylabel('Number of steps')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

        plt.suptitle(
            'sd1 (exponential cooling) vs sd2 (slope-adaptive) — Simulated Annealing',
            fontsize=13, y=1.01
        )
        plt.tight_layout()
        plt.savefig('experiment_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("  Saved → experiment_results.png")

    def run_experiment(self, problem):
        """Executes the complete testing suite and generates all reports and figures."""
        print(f"Running {self.experiment_time} runs x {self.loop_time} steps …")
        results_sd1 = self.multiple_run(problem, SCHEDULE_FUNCTION_1)
        results_sd2 = self.multiple_run(problem, SCHEDULE_FUNCTION_2)

        print("Running single-run samples for convergence chart …")
        z_hist1 = self.single_run(problem, SCHEDULE_FUNCTION_1)
        z_hist2 = self.single_run(problem, SCHEDULE_FUNCTION_2)

        self.generate_summary_table(results_sd1, results_sd2)
        self.generate_figures(results_sd1, results_sd2, z_hist1, z_hist2)


if __name__ == '__main__':
    prob = Problem(IMAGE_PATH)
    exper = Tester(EXPERIMENT, LOOP)
    exper.run_experiment(prob)