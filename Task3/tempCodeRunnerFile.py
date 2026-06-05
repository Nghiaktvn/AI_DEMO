import sys
from problem import CSPProblem
from solver import CSPSolver
from visualizer import Visualizer


DEMO_MATRIX = [
    [None, 2,    3,    None, None, 0,    None, None, None, None],
    [None, None, None, None, 3,    None, 2,    None, None, 6   ],
    [None, None, 5,    None, 5,    3,    None, 5,    7,    4   ],
    [None, 4,    None, 5,    None, 5,    None, 6,    None, 3   ],
    [None, None, 4,    None, 5,    None, 6,    None, None, 3   ],
    [None, None, None, 2,    None, 5,    None, None, None, None],
    [4,    None, 1,    None, None, None, 1,    1,    None, None],
    [4,    None, 1,    None, None, None, 1,    None, 4,    None],
    [None, None, None, None, 6,    None, None, None, None, 4   ],
    [None, 4,    4,    None, None, None, None, 4,    None, None],
]


def load_problem(path: str) -> CSPProblem:
    return CSPProblem.load_from_file(path)


def run(problem: CSPProblem, show_constraints: bool = True) -> None:
    viz = Visualizer(problem)

    if show_constraints:
        for line in problem.describe_constraints():
            print(line)

    print("\n=== Encoding CSP to CNF ===")
    csp_solver = CSPSolver(problem)

    print("Solving with Glucose3 (PySAT)...")
    sat = csp_solver.solve()

    print(f"Variables : {csp_solver.variable_count()}")
    print(f"Clauses   : {csp_solver.clause_count()}")

    if sat:
        print("Result    : SATISFIABLE")
        print("\nDisplaying solution...")
        viz.show(csp_solver.solution, save_path="solution.png")
    else:
        print("Result    : UNSATISFIABLE — no valid coloring exists.")
        viz.show(save_path="input_only.png")


def main() -> None:
    args = sys.argv[1:]

    constraints_only = "--constraints-only" in args
    args = [a for a in args if a != "--constraints-only"]

    if args:
        problem = load_problem(args[0])
        print(f"Loaded matrix from '{args[0]}' "
              f"({problem.rows}x{problem.cols})")
    else:
        problem = CSPProblem(DEMO_MATRIX)
        print(f"Using built-in demo matrix ({problem.rows}x{problem.cols})")

    run(problem, show_constraints=True)


if __name__ == "__main__":
    main()