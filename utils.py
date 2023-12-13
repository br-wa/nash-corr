import numpy as np
import gurobipy as gp
from gurobipy import GRB

def nash_support(A, B, N, gurobienv, debug=False):
    # computes the support size of the smallest Nash equilibrium with utilty matrices A and B
    assert A.shape == (N, N)
    assert B.shape == (N, N)    

    Ua = np.max(A) - np.min(A)
    Ub = np.max(B) - np.min(B)
    ones = np.full(N, 1)

    with gp.Model("game", env=gurobienv) as model:
        x = model.addMVar(N, lb=0)
        xignored = model.addMVar(N, vtype=GRB.BINARY)
        y = model.addMVar(N, lb=0)
        yignored = model.addMVar(N, vtype=GRB.BINARY)
        vA = model.addVar(lb=-float('inf'))
        vB = model.addVar(lb=-float('inf'))
    
        model.addConstr(A @ y <= vA)
        model.addConstr(B.T @ x <= vB)
        model.addConstr(A @ y >= vA - Ua * xignored)
        model.addConstr(B.T @ x >= vB - Ub * yignored)
        model.addConstr(ones @ x == 1)
        model.addConstr(ones @ y == 1)
        model.addConstr(x + xignored <= 1)
        model.addConstr(y + yignored <= 1)
    
        model.setObjective(ones @ (xignored + yignored), GRB.MAXIMIZE)

        model.optimize()
    
        if debug == True:
            print(x)
            print(y)

        if model.status == GRB.OPTIMAL:
             sol_size = (N - model.objVal / 2)
        else:
            sol_size = 0

    return sol_size

if __name__ == "__main__":
    with gp.Env() as env:
        env.setParam("OutputFlag", 0)
        env.start()
        A = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
        B = -A
        print(nash_support(A, B, 3, env, debug=True))
