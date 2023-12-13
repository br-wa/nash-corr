import argparse
from utils import nash_support
import gurobipy as gp
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, required=True)
parser.add_argument("--N_ITERS", type=int, required=True)
parser.add_argument("--rho", type=float, required=True)
parser.add_argument("--output_file", required=True)
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()

N = args.N
rho = args.rho
N_ITERS = args.N_ITERS
output_file = args.output_file
batch_size = args.batch_size

n_feasible = 0
n_majorized = 0

env = gp.Env()
env.setParam("OutputFlag", 0)
env.start()

results = []

for _ in tqdm(range(N_ITERS)):
    A = [np.zeros((N, N)) for _ in batch_size]
    B = [np.zeros((N, N)) for _ in batch_size]

    for k in range(batch_size):
        for i in range(N):
            for j in range(N):
                # sample x, y normals with mean 0 variance 1 and covariance rho
                [[x, y]] = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 1)
                A[k][i, j] = x
                B[k][i, j] = y
        
    result = nash_support(A, B, N, env, batch_size=batch_size)
    results.append(result)

np.save(output_file, np.array(results))
