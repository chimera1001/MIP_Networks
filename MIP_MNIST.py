# CNN MIP Model for MNIST with Max Pooling, Flattening, and Diagnostics
import pyomo.environ as pyo
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load MNIST data (binary classification: digit 3)
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.reshape(-1, 28, 28).astype(np.float32) / 255.0
y = (y.astype(int) == 3).astype(int)
X, y = X[:100], y[:100]  # tractable subset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)

# Configuration
n, H, W = X_train.shape[0], 28, 28
K, stride = 5, 2
C_out = 2
H_out, W_out = (H - K) // stride + 1, (W - K) // stride + 1
P = 2
H_pool, W_pool = H_out // P, W_out // P
F = C_out * H_pool * W_pool
M = 5.0
alpha, beta, lambda_ = 0.5, 10, 0.9

model = pyo.ConcreteModel()
model.I = range(n)
model.Co = range(C_out)
model.K = range(K)
model.Ho = range(H_out)
model.Wo = range(W_out)
model.Hp = range(H_pool)
model.Wp = range(W_pool)
model.P = range(P)
model.F = range(F)

model.W = pyo.Var(model.Co, model.K, model.K, bounds=(-M, M))
model.b = pyo.Var(model.Co, bounds=(-M, M))
model.z = pyo.Var(((i, c, h, w) for i in model.I for c in model.Co for h in model.Ho for w in model.Wo), domain=pyo.Reals)
model.a = pyo.Var(((i, c, h, w) for i in model.I for c in model.Co for h in model.Ho for w in model.Wo), bounds=(0, M))
model.delta = pyo.Var(((i, c, h, w) for i in model.I for c in model.Co for h in model.Ho for w in model.Wo), domain=pyo.Binary)
model.pool = pyo.Var(((i, c, hp, wp) for i in model.I for c in model.Co for hp in model.Hp for wp in model.Wp), bounds=(0, M))
model.zeta = pyo.Var(((i, c, hp, wp, ph, pw) for i in model.I for c in model.Co for hp in model.Hp for wp in model.Wp for ph in model.P for pw in model.P), domain=pyo.Binary)
model.feat = pyo.Var(((i, f) for i in model.I for f in model.F), domain=pyo.Reals)
model.Wfc = pyo.Var(model.F, bounds=(-M, M))
model.bfc = pyo.Var(bounds=(-M, M))
model.pred = pyo.Var(model.I, domain=pyo.Reals)
model.u = pyo.Var(model.Co, model.K, model.K, domain=pyo.NonNegativeReals)
model.gamma = pyo.Var(model.Co, domain=pyo.Binary)

# Constraints
model.conv = pyo.ConstraintList()
for i in model.I:
    for c in model.Co:
        for h in model.Ho:
            for w in model.Wo:
                s_h, s_w = h * stride, w * stride
                patch_sum = sum(model.W[c, u, v] * float(X_train[i][s_h + u][s_w + v]) for u in model.K for v in model.K)
                model.conv.add(model.z[i, c, h, w] == patch_sum + model.b[c])

model.relu = pyo.ConstraintList()
for i in model.I:
    for c in model.Co:
        for h in model.Ho:
            for w in model.Wo:
                model.relu.add(model.a[i, c, h, w] >= model.z[i, c, h, w])
                model.relu.add(model.a[i, c, h, w] >= 0)
                model.relu.add(model.a[i, c, h, w] <= model.z[i, c, h, w] + M * (1 - model.delta[i, c, h, w]))
                model.relu.add(model.a[i, c, h, w] <= M * model.delta[i, c, h, w])

model.pooling = pyo.ConstraintList()
for i in model.I:
    for c in model.Co:
        for hp in model.Hp:
            for wp in model.Wp:
                model.pooling.add(sum(model.zeta[i, c, hp, wp, ph, pw] for ph in model.P for pw in model.P) == 1)
                for ph in model.P:
                    for pw in model.P:
                        h_full = hp * P + ph
                        w_full = wp * P + pw
                        model.pooling.add(model.pool[i, c, hp, wp] >= model.a[i, c, h_full, w_full])
                        model.pooling.add(model.pool[i, c, hp, wp] <= model.a[i, c, h_full, w_full] + M * (1 - model.zeta[i, c, hp, wp, ph, pw]))

model.flatten = pyo.ConstraintList()
for i in model.I:
    for c in model.Co:
        for hp in model.Hp:
            for wp in model.Wp:
                f = c * H_pool * W_pool + hp * W_pool + wp
                model.flatten.add(model.feat[i, f] == model.pool[i, c, hp, wp])

def output_rule(m, i):
    return m.pred[i] == sum(m.Wfc[f] * m.feat[i, f] for f in model.F) + m.bfc
model.output = pyo.Constraint(model.I, rule=output_rule)

model.prune = pyo.ConstraintList()
model.absW = pyo.ConstraintList()
for c in model.Co:
    for u in model.K:
        for v in model.K:
            model.prune.add(model.W[c, u, v] <= M * model.gamma[c])
            model.prune.add(model.W[c, u, v] >= -M * model.gamma[c])
            model.absW.add(model.u[c, u, v] >= model.W[c, u, v])
            model.absW.add(model.u[c, u, v] >= -model.W[c, u, v])

model.obj = pyo.Objective(
    expr=sum((model.pred[i] - y_train[i])**2 for i in model.I)
         + alpha * lambda_ * sum(model.u[c, u, v] for c in model.Co for u in model.K for v in model.K)
         + 0.5 * alpha * (1 - lambda_) * sum(model.W[c, u, v]**2 for c in model.Co for u in model.K for v in model.K)
         + beta * sum(model.gamma[c] for c in model.Co),
    sense=pyo.minimize
)

# Solve
solver = pyo.SolverFactory("gurobi")
solver.options["TimeLimit"] = 60
results = solver.solve(model, tee=True)

HIDDEN_DIM = 10
F = len(model.F)

# Extract trained weights
W_hidden = np.array([[pyo.value(model.W_hidden[j, f]) for f in model.F] for j in range(HIDDEN_DIM)])
b_hidden = np.array([pyo.value(model.b_hidden[j]) for j in range(HIDDEN_DIM)])
W_out = np.array([pyo.value(model.W_out[j]) for j in range(HIDDEN_DIM)])
b_out = pyo.value(model.b_out)

def relu(x): return np.maximum(0, x)

def conv_layer(img, W, b, stride=2):
    C_out, K, _ = W.shape
    H, W_ = img.shape
    H_out, W_out = (H - K) // stride + 1, (W_ - K) // stride + 1
    out = np.zeros((C_out, H_out, W_out))
    for c in range(C_out):
        for h in range(H_out):
            for w in range(W_out):
                patch = img[h*stride:h*stride+K, w*stride:w*stride+K]
                out[c, h, w] = relu(np.sum(W[c] * patch) + b[c])
    return out

def max_pool_layer(A, pool_size=2):
    C, H, W = A.shape
    Hp, Wp = H // pool_size, W // pool_size
    pooled = np.zeros((C, Hp, Wp))
    for c in range(C):
        for h in range(Hp):
            for w in range(Wp):
                pooled[c, h, w] = np.max(A[c, h*pool_size:(h+1)*pool_size, w*pool_size:(w+1)*pool_size])
    return pooled

def flatten(A): return A.flatten()

# Inference loop
preds = []
for x in X_test:
    A1 = conv_layer(x, W_val, b_val)
    A2 = max_pool_layer(A1)
    flat = flatten(A2)
    hidden = relu(np.dot(W_hidden, flat) + b_hidden)
    output = np.dot(W_out, hidden) + b_out
    preds.append(1 if output >= 0.5 else 0)

acc = accuracy_score(y_test, preds)
print(f"✅ Test Accuracy: {acc:.3f}")

# Filter sparsity analysis
sparsity = {}
for c in model.Co:
    flat_weights = [pyo.value(model.W[c, u, v]) for u in model.K for v in model.K]
    total = len(flat_weights)
    nonzero = np.count_nonzero(flat_weights)
    sparsity[c] = {"nonzero": nonzero, "total": total, "sparsity": 1 - nonzero / total}

# Save diagnostics
diagnostics = {
    "filters": W_val,
    "filter_biases": b_val,
    "W_fc": Wfc_val,
    "b_fc": bfc_val,
    "filter_retention": gamma_vals,
    "relu_activity_map": relu_activity,
    "accuracy": acc,
    "filter_sparsity": sparsity
}

with open("cnn_mip_diagnostics.pkl", "wb") as f:
    pickle.dump(diagnostics, f)