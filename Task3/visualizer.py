import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from problem import CSPProblem
class Visualizer:
    """
    Renders the CSP input matrix and the colored solution side-by-side
    using matplotlib.
    GREEN cells  -> bright green (#4CAF50)
    RED cells    -> crimson     (#E53935)
    Blank cells  -> light gray  (#EEEEEE)
    """
    GREEN = "#4CAF50"
    RED   = "#E53935"
    BLANK =  "#EEEEEE"
    TEXT_DARK  ="#1A1A1A"
    TEXT_LIGHT ="#FFFFFF"

    def __init__(self, problem: CSPProblem):
        self.problem = problem

    def _cell_color(self, r: int, c: int,solution: list[list[bool]] | None) -> str:
        if solution is None:
            return self.BLANK
        return self.GREEN if solution[r][c] else self.RED

    def _draw_grid(self, ax, solution: list[list[bool]] | None,
                   title: str) -> None:
        rows, cols = self.problem.rows, self.problem.cols
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13, pad=8)

        for r in range(rows):
            for c in range(cols):
                color = self._cell_color(r, c, solution)
                rect = mpatches.FancyBboxPatch(
                    (c + 0.04, rows - r - 1 + 0.04),
                    0.92, 0.92,
                    boxstyle="round,pad=0.05",
                    facecolor=color,
                    edgecolor="#555555",
                    linewidth=0.6,
                )
                ax.add_patch(rect)

                val = self.problem.get_value(r, c)
                if val is not None:
                    txt_color = (self.TEXT_LIGHT
                                 if color in (self.GREEN, self.RED)
                                 else self.TEXT_DARK)
                    ax.text(
                        c + 0.5, rows - r - 0.5, str(val),
                        ha="center", va="center",
                        fontsize=max(7, min(14, 120 // max(rows, cols))),
                        fontweight="bold",
                        color=txt_color,
                    )

    def show(self, solution: list[list[bool]] | None = None,
             save_path: str | None = None) -> None:
        """
        Display input matrix and (optionally) the colored solution.
        If solution is None, only the input grid is shown.
        """
        n_plots = 1 if solution is None else 2
        fig, axes = plt.subplots(1, n_plots,
                                 figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        self._draw_grid(axes[0], None, "Input matrix")
        if solution is not None:
            self._draw_grid(axes[1], solution, "Solution")

        green_p = mpatches.Patch(color=self.GREEN, label="GREEN")
        red_p   = mpatches.Patch(color=self.RED,   label="RED")
        blank_p = mpatches.Patch(color=self.BLANK,  label="Blank")
        fig.legend(handles=[green_p, red_p, blank_p],
                   loc="lower center", ncol=3, fontsize=10,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)