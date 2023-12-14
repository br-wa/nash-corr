import numpy as np
import gurobipy as gp
from gurobipy import GRB

def nash_support(A, B, N, gurobienv, cap=None, batch_size=1, debug=False):
    # computes the support size of the smallest Nash equilibrium with utilty matrices A and B
    Ua = max([np.max(Ai) for Ai in A]) - min([np.min(Ai) for Ai in A])
    Ub = max([np.max(Bi) for Bi in B]) - min([np.min(Bi) for Bi in B])
    ones = np.full(N, 1)
    if cap is None:
        cap = N

    with gp.Model("game", env=gurobienv) as model:
        x = []
        xignored = []
        y = []
        yignored = []
        vA = []
        vB = []
        for _ in range(batch_size):
            x.append(model.addMVar(N, lb=0))
            xignored.append(model.addMVar(N, vtype=GRB.BINARY))
            y.append(model.addMVar(N, lb=0))
            yignored.append(model.addMVar(N, vtype=GRB.BINARY))
            vA.append(model.addVar(lb=-float('inf')))
            vB.append(model.addVar(lb=-float('inf')))
    
        for i in range(batch_size):
            model.addConstr(A[i] @ y[i] <= vA[i])
            model.addConstr(B[i].T @ x[i] <= vB[i])
            model.addConstr(A[i] @ y[i] >= vA[i] - Ua * xignored[i])
            model.addConstr(B[i].T @ x[i] >= vB[i] - Ub * yignored[i])
            model.addConstr(ones @ x[i] == 1)
            model.addConstr(ones @ y[i] == 1)
            model.addConstr(x[i] + xignored[i] <= 1)
            model.addConstr(y[i] + yignored[i] <= 1)
            model.addConstr(ones @ xignored[i] >= N-cap)
            model.addConstr(ones @ yignored[i] >= N-cap)
        
        local_sum = [ones @ (xignored[i] + yignored[i]) for i in range(batch_size)]
        model.setObjective(sum(local_sum), GRB.MAXIMIZE)

        model.optimize()
    
        if debug == True:
            print(x)
            print(y)

        if model.status == GRB.OPTIMAL:
            sol_sizes = []
            for i in range(batch_size):
                sol_sizes.append(N - sum([xi.X for xi in xignored[i]]))
                
        else:
            sol_sizes = [0 for _ in range(batch_size)]

    return sol_sizes

if __name__ == "__main__":
    with gp.Env() as env:
        env.start()
        A = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
        B = -A
        print(nash_support([A, A], [B, B], 3, env, batch_size=2, debug=True))
        print(nash_support([A, A], [B, B], 3, env, cap=2, batch_size=2, debug=True))
