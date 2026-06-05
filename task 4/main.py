import graphviz
from collections import deque
import time
from IPython.display import Image, display

# create data file
data = """-p q
-l -m p
-b -l m
-a -p l
-a -b l
a
b
"""
with open("kb.txt", "w") as f:
    f.write(data)
#Class Clause
class Clause:
    def __init__(self, line):
        parts = line.split()
        self._premises = [] 
        self._conclusion = ""

        for p in parts:
            if p.startswith("-"):
                self._premises.append(p[1:])
            else:
                self._conclusion = p
        self._premises = tuple(self._premises)
        
    def get_premises(self):
        return self._premises

    def get_conclusion(self):
        return self._conclusion

    def is_fact(self):
        if len(self._premises) == 0:
            return True
        else:
            return False

    def has_premise(self, s):
        for p in self._premises:
            if p == s:
                return True
        return False
        
    def __hash__(self):
        return hash((self._premises, self._conclusion))

    def __eq__(self, other):
        return isinstance(other, Clause) and \
               self._premises == other._premises and \
               self._conclusion == other._conclusion
    def __str__(self):
        if self.is_fact():
            return "FACT: " + self._conclusion

        res = ""
        for i in range(len(self._premises)):
            res += self._premises[i]
            if i < len(self._premises) - 1:
                res += " ^ "
        return res + " => " + self._conclusion
        
# class knowledgebase 
class KnowledgeBase:
    def __init__(self, filepath):
        self._clauses = []
        self._symbols = set()
        self._rules_by_head = {}
        self._load(filepath)

    def _load(self, filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    c = Clause(line)
                    self._clauses.append(c)

                    self._symbols.add(c.get_conclusion())
                    self._symbols.update(c.get_premises())

                    self._rules_by_head.setdefault(c.get_conclusion(), []).append(c)

    # Safe access
    def get_symbols(self):
        return tuple(sorted(self._symbols)) 
    def get_facts(self):
        return tuple(c.get_conclusion() for c in self._clauses if c.is_fact())

    def get_all_clauses(self):
        return tuple(self._clauses)

    def get_clauses_with_premise(self, symbol):
        return tuple(c for c in self._clauses if c.has_premise(symbol))

    def get_clauses_for_goal(self, goal):
        return tuple(self._rules_by_head.get(goal, []))

    # Draw graph
    def draw_graph(self):
        dot = graphviz.Digraph(format="png")
        dot.attr(rankdir='LR')
        dot.attr(dpi="100")
        
        facts = set(self.get_facts())

        for s in self.get_symbols():
            if s in facts:
                dot.node(s, s, style="filled", fillcolor="#d4edda", fontname="Arial Bold")
            else:
                dot.node(s, s, fontname="Arial")

        and_id = 0
        for c in self.get_all_clauses():
            if not c.is_fact():
                premises = c.get_premises()
                conclusion = c.get_conclusion()

                if len(premises) > 1:
                    a = f"and_{and_id}"
                    and_id += 1
                    dot.node(a, "AND", shape="circle", fontsize="10", 
                             width="0.8", height="0.8", fixedsize="true",
                             style="filled", fillcolor="#f8f9fa")

                    for p in premises:
                        dot.edge(p, a, arrowhead="none")
                    dot.edge(a, conclusion)
                else:
                    dot.edge(premises[0], conclusion)

        path = dot.render("kb_graph", cleanup=True)
        display(Image(path))

# class inference engine
class InferenceEngine:
    def __init__(self, kb: KnowledgeBase):
        self._kb = kb

    # Forward chaining
    def forward(self, query):
        clauses = self._kb.get_all_clauses()

        count = {c: len(c.get_premises()) for c in clauses}
        inferred = {s: False for s in self._kb.get_symbols()}
        agenda = deque(self._kb.get_facts())
        order = []

        while agenda:
            p = agenda.popleft()

            if p == query:
                if p not in order:
                    order.append(p)
                return True, order
            if not inferred[p]:
                inferred[p] = True
                order.append(p)

                for c in self._kb.get_clauses_with_premise(p):
                    count[c] -= 1
                    if count[c] == 0:
                        agenda.append(c.get_conclusion())
        return False, order
    # Backward chaining
    def backward(self, query):
        inferred = []
        proven = set()
        visiting = set()

        def solve(goal):
            if goal in proven:
                return True
            if goal in visiting:
                return False
            visiting.add(goal)

            for c in self._kb.get_clauses_for_goal(goal):
                if c.is_fact() or all(solve(p) for p in c.get_premises()):
                    proven.add(goal)
                    if goal not in inferred:
                        inferred.append(goal)
                    visiting.remove(goal)
                    return True

            visiting.remove(goal)
            return False
        return solve(query), inferred

    #experiment
    def run_experiment(self):
        results = []

        for s in self._kb.get_symbols():
            t1 = time.perf_counter()
            fc_res, f_nodes = self.forward(s)
            t2 = time.perf_counter()

            t3 = time.perf_counter()
            bc_res, b_nodes = self.backward(s)
            t4 = time.perf_counter()

            results.append({
                "query": s,
                "result": "YES" if fc_res else "NO",
                "fc_nodes": len(f_nodes),
                "bc_nodes": len(b_nodes),
                "diff": len(f_nodes) - len(b_nodes),
                "fc_time": round((t2 - t1) * 1e6, 1),
                "bc_time": round((t4 - t3) * 1e6, 1)
            })
        self._print(results)
    def _print(self, results):
        print("\n" + "="*80)
        print("EXPERIMENTAL RESULTS: Forward vs Backward Chaining".center(80))
        print("="*80)
        print(f"{'Query':<8}{'Result':<8}{'FC Nodes':>10}{'BC Nodes':>10}{'Diff':>8}{'FC(µs)':>12}{'BC(µs)':>12}")
        print("-"*80)

        for r in results:
            print(f"{r['query']:<8}{r['result']:<8}{r['fc_nodes']:>10}{r['bc_nodes']:>10}{r['diff']:>8}{r['fc_time']:>12}{r['bc_time']:>12}")

        total_fc = sum(r["fc_nodes"] for r in results)
        total_bc = sum(r["bc_nodes"] for r in results)

        print("-"*80)
        print(f"{'Total':<16}{total_fc:>10}{total_bc:>10}{total_fc-total_bc:>8}")

if __name__ == "__main__":
    print("REQUIREMENT 1: GRAPH")
    kb = KnowledgeBase("kb.txt")
    for clause in kb.get_all_clauses():
        print(str(clause))
    kb.draw_graph()
    engine = InferenceEngine(kb)

    print("REQUIREMENT 2 & 3: INFERENCE")
    q_fc = input("Enter the symbol to verify for Forward Chaining (EX: 'q'): ").strip().lower()
    fc_res, fc_nodes = engine.forward(q_fc)
    q_bc = input("Enter the symbol to verify for Backward Chaining (EX:'q'): ").strip().lower()
    bc_res, bc_nodes = engine.backward(q_bc)
    print(f"Forward Chaining ({q_fc}): Result = {fc_res}, Visited Nodes = {fc_nodes}")
    print(f"Backward Chaining ({q_bc}): Result = {bc_res}, Visited Nodes = {bc_nodes}")

    print("\nREQUIREMENT 4: EXPERIMENTAL COMPARISON")
    engine.run_experiment()