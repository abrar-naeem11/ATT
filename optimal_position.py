import pulp
import numpy as np
from scipy.cluster.vq import kmeans2

# Common Parameters
L = 1000.0  # environment size L x L in meters
Rcov = 202.07
M = 2000.0
max_users_per_uav = 20
num_hotspots = 5
rhotspot = 200.0
np.random.seed(42)  # For reproducibility

# Function to generate users
def generate_users(num_users):
    hotspots = np.random.uniform(0, L, size=(num_hotspots, 2))
    D = num_users // num_hotspots
    users = []
    for tx, ty in hotspots:
        for _ in range(D):
            r = np.random.uniform(0, rhotspot)
            phi = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(phi) + tx
            y = r * np.sin(phi) + ty
            users.append([x, y])
    # Remaining users randomly distributed
    remaining = num_users - len(users)
    for _ in range(remaining):
        x = np.random.uniform(0, L)
        y = np.random.uniform(0, L)
        users.append([x, y])
    return np.array(users)

# Linear approximation parameters for circle
num_approx = 8  # octagon approximation
alphas = np.linspace(0, 2 * np.pi, num_approx, endpoint=False)
cos_pi_n = np.cos(np.pi / num_approx)
R_approx = Rcov * cos_pi_n

# Direct MILP Implementation (for small number of users)
def direct_milp(num_users, num_uavs):
    users = generate_users(num_users)
    n = len(users)
    prob = pulp.LpProblem("UAV_Positioning_Direct", pulp.LpMaximize)
    
    # Variables
    x_uav = pulp.LpVariable.dicts("x_uav", range(num_uavs), lowBound=Rcov, upBound=L - Rcov)
    y_uav = pulp.LpVariable.dicts("y_uav", range(num_uavs), lowBound=Rcov, upBound=L - Rcov)
    X = pulp.LpVariable.dicts("X", (range(num_uavs), range(n)), cat='Binary')
    
    # Objective
    prob += pulp.lpSum(X[i][j] for i in range(num_uavs) for j in range(n))
    
    # Constraints
    for j in range(n):
        prob += pulp.lpSum(X[i][j] for i in range(num_uavs)) <= 1
    for i in range(num_uavs):
        prob += pulp.lpSum(X[i][j] for j in range(n)) <= max_users_per_uav
    
    # Distance constraints (linear approx)
    for i in range(num_uavs):
        for j in range(n):
            xu, yu = users[j]
            for k in range(num_approx):
                cos_a = np.cos(alphas[k])
                sin_a = np.sin(alphas[k])
                left = cos_a * x_uav[i] + sin_a * y_uav[i]
                right = R_approx + cos_a * xu + sin_a * yu
                prob += left <= right + M * (1 - X[i][j])
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Results
    status = pulp.LpStatus[prob.status]
    optimal_value = pulp.value(prob.objective)
    uav_positions = [(pulp.value(x_uav[i]), pulp.value(y_uav[i])) for i in range(num_uavs)]
    return status, optimal_value, uav_positions

# Clustering-based MILP Implementation (for large number of users)
def clustering_milp(num_users, num_uavs, num_clusters):
    users = generate_users(num_users)
    centroids, labels = kmeans2(users, num_clusters, minit='points')
    Uj = np.bincount(labels, minlength=num_clusters)
    
    prob = pulp.LpProblem("UAV_Positioning_Clustering", pulp.LpMaximize)
    
    # Variables
    x_uav = pulp.LpVariable.dicts("x_uav", range(num_uavs), lowBound=Rcov, upBound=L - Rcov)
    y_uav = pulp.LpVariable.dicts("y_uav", range(num_uavs), lowBound=Rcov, upBound=L - Rcov)
    Y = pulp.LpVariable.dicts("Y", (range(num_uavs), range(num_clusters)), lowBound=0, upBound=1)
    epsilon = pulp.LpVariable.dicts("epsilon", (range(num_uavs), range(num_clusters)), cat='Binary')
    
    # Objective
    prob += pulp.lpSum(Y[i][j] * Uj[j] for i in range(num_uavs) for j in range(num_clusters))
    
    # Constraints
    for i in range(num_uavs):
        prob += pulp.lpSum(Y[i][j] * Uj[j] for j in range(num_clusters)) <= max_users_per_uav
    for j in range(num_clusters):
        prob += pulp.lpSum(Y[i][j] for i in range(num_uavs)) <= 1
    for i in range(num_uavs):
        for j in range(num_clusters):
            prob += Y[i][j] <= 1 - epsilon[i][j]
            xc, yc = centroids[j]
            for k in range(num_approx):
                cos_a = np.cos(alphas[k])
                sin_a = np.sin(alphas[k])
                left = cos_a * x_uav[i] + sin_a * y_uav[i]
                right = R_approx + cos_a * xc + sin_a * yc
                prob += left <= right + M * epsilon[i][j]
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Results
    status = pulp.LpStatus[prob.status]
    optimal_value = pulp.value(prob.objective)
    uav_positions = [(pulp.value(x_uav[i]), pulp.value(y_uav[i])) for i in range(num_uavs)]
    return status, optimal_value, uav_positions

# Example usage for direct MILP (small scale, as in paper Section 6.1)
print("Direct MILP Example (100 users, 5 UAVs):")
status, opt_val, positions = direct_milp(100, 5)
print("Status:", status)
print("Optimal Value (Users Covered):", opt_val)
print("UAV Positions:")
for i, pos in enumerate(positions):
    print(f"UAV {i}: {pos}")

# Example usage for clustering MILP (large scale, as in paper Section 6.2)
print("\nClustering MILP Example (1000 users, 10 UAVs, 10 clusters):")
status, opt_val, positions = clustering_milp(1000, 10, 10)
print("Status:", status)
print("Optimal Value (Users Covered):", opt_val)
print("UAV Positions:")
for i, pos in enumerate(positions):
    print(f"UAV {i}: {pos}")
