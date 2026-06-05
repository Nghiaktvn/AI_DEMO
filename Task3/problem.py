class CSPProblem:
    """
    Represents the CSP coloring puzzle.
    Each cell is either blank (None) or holds an integer k >= 0.
    Variables: each cell (i, j) -> domain {True=GREEN, False=RED}.
    Constraint: for each numbered cell, exactly k of its 9 neighbors are green.
    """

    def __init__(self, matrix: list[list]):
        """
        matrix: 2D list of int or None (None = blank cell).
        """
        self.matrix = matrix
        self.rows = len(matrix)
        self.cols = len(matrix[0]) if self.rows > 0 else 0

    def get_value(self, r: int, c: int):
        """Return the cell number or None if blank."""
        return self.matrix[r][c]

    def is_blank(self, r: int, c: int) -> bool:
        return self.matrix[r][c] is None

    def neighbors(self, r: int, c: int) -> list[tuple[int, int]]:
        """
        Return all valid neighbors of (r, c) including itself (9-adjacency).
        Cells at borders have fewer neighbors.
        """
        result = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    result.append((nr, nc))
        return result

    def numbered_cells(self) -> list[tuple[int, int, int]]:
        """Return list of (r, c, k) for every numbered cell."""
        result = []
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.is_blank(r, c):
                    result.append((r, c, self.matrix[r][c]))
        return result

    def describe_constraints(self) -> list[str]:
        """
        Return human-readable descriptions of all constraints.
        Used for Requirement 1.
        """
        lines = []
        lines.append(" CSP Constraints ")
        lines.append("Domain: each cell in {GREEN=True, RED=False}")
        lines.append("")
        lines.append("Type 1 – Numbered cell constraint:")
        lines.append("  For each cell (r,c) with value k:")
        lines.append("  SUM of GREEN neighbors (incl. itself) == k")
        lines.append("")
        lines.append("Type 2 – Border adjustment:")
        lines.append("  Cells near borders have fewer than 9 neighbors;")
        lines.append("  k cannot exceed the actual neighbor count.")
        lines.append("")
        lines.append("Type 3 – Blank cells:")
        lines.append("  No constraint imposed; color can be anything.")
        lines.append("")
        lines.append("Examples from current matrix:")
        for r, c, k in self.numbered_cells()[:5]:
            nbrs = self.neighbors(r, c)
            lines.append(f"  Cell ({r},{c})=={k}: {len(nbrs)} neighbors -> "
                         f"exactly {k} must be GREEN")
        return lines

    @staticmethod
    def load_from_file(path: str) -> "CSPProblem":
        """
        Load matrix from a text file.
        Each row is space-separated; '.' means blank cell.
        """
        matrix = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = []
                for token in line.split():
                    row.append(None if token == '.' else int(token))
                matrix.append(row)
        return CSPProblem(matrix)