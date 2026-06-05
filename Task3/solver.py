from pysat.solvers import Glucose3
from encoder import CNFEncoder
from problem import CSPProblem


class CSPSolver:
    """
    Wraps PySAT Glucose3 to solve the encoded CSP.
    """

    def __init__(self, problem: CSPProblem):
        self.problem = problem
        self.encoder = CNFEncoder(problem)
        self._solution: list[list[bool]] | None = None
        self._sat: bool = False

    def solve(self) -> bool:
        """
        Encode and solve the CSP.
        Returns True if satisfiable, False otherwise.
        """
        clauses = self.encoder.encode()
        n_vars = self.encoder.num_variables()

        solver = Glucose3()
        solver.append_formula(clauses)
        self._sat = solver.solve()
        if self._sat:
            model = solver.get_model()
            self._solution = self.encoder.decode(model)

        solver.delete()
        return self._sat

    @property
    def is_satisfiable(self) -> bool:
        return self._sat

    @property
    def solution(self) -> list[list[bool]] | None:
        """2D grid: True=GREEN, False=RED. None if unsatisfiable."""
        return self._solution

    def clause_count(self) -> int:
        return len(self.encoder.clauses)

    def variable_count(self) -> int:
        return self.encoder.num_variables() 