from itertools import combinations
from problem import CSPProblem

class CNFEncoder:
    """
    Variable numbering (1-indexed, required by PySAT):
        var(r, c) = r * cols + c + 1
    For each numbered cell (r, c) with value k and neighbor list N:
        Exactly-k(N) = at-least-k(N) AND at-most-k(N)
    at-least-k(N, k):
        For every subset S of N with |S| = |N|-k+1,
        at least one cell in S must be GREEN.
        => Clause: (OR of vars in S)
        Explanation: if fewer than k cells were green, some (|N|-k+1)-subset
        would be all red — this clause blocks that.
    at-most-k(N, k):
        For every subset S of N with |S| = k+1,s
        at least one cell in S must be RED (not all green).
        => Clause: (OR of NEG vars in S)
    """

    def __init__(self, problem: CSPProblem):
        self.problem = problem
        self.rows = problem.rows
        self.cols = problem.cols
        self.clauses: list[list[int]] = []
        self._clause_set: set[frozenset] = set()

    def var(self, r: int, c: int) -> int:
        """Map cell (r,c) to a positive integer variable for PySAT."""
        return r * self.cols + c + 1

    def _add_clause(self, clause: list[int]) -> None:
        """Add a clause if not already present (no duplicates)."""
        key = frozenset(clause)
        if key not in self._clause_set:
            self._clause_set.add(key)
            self.clauses.append(clause)

    def _encode_at_least(self, var_list: list[int], k: int) -> None:
        """
        at-least-k: for every (|N|-k+1)-subset of var_list,
        add clause requiring at least one positive literal.
        """
        n = len(var_list)
        size = n - k + 1
        if size <= 0:
            return
        for subset in combinations(var_list, size):
            self._add_clause(list(subset))

    def _encode_at_most(self, var_list: list[int], k: int) -> None:
        """
        at-most-k: for every (k+1)-subset of var_list,
        add clause requiring at least one negative literal.
        """
        for subset in combinations(var_list, k + 1):
            self._add_clause([-v for v in subset])

    def encode(self) -> list[list[int]]:
        """
        Generate all CNF clauses for the problem.
        Returns list of clauses (each clause is list of ints).
        """
        self.clauses = []
        self._clause_set = set()

        for r, c, k in self.problem.numbered_cells():
            nbrs = self.problem.neighbors(r, c)
            var_list = [self.var(nr, nc) for (nr, nc) in nbrs]
            n = len(var_list)

            if k < 0 or k > n:
                raise ValueError(
                    f"Cell ({r},{c}) has value {k} but only {n} neighbors.")

            self._encode_at_least(var_list, k)
            self._encode_at_most(var_list, k)

        return self.clauses

    def num_variables(self) -> int:
        return self.rows * self.cols

    def decode(self, model: list[int]) -> list[list[bool]]:
        """
        Convert PySAT model (list of signed ints) to a 2D bool grid.
        True  -> GREEN
        False -> RED
        """
        true_vars = set(v for v in model if v > 0)
        grid = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                row.append(self.var(r, c) in true_vars)
            grid.append(row)
        return grid