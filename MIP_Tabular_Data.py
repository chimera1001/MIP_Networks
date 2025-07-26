import pyomo.environ as pyo
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

# **Step 1: Configure the Dataset**
dataset_name = 'breast_cancer'  # Options: 'iris', 'breast_cancer', 'wine'

# Load and prepare the dataset
if dataset_name == 'iris':
    data = load_iris()
    n_classes = 3
elif dataset_name == 'breast_cancer':
    data = load_breast_cancer()
    n_classes = 2
elif dataset_name == 'wine':
    data = load_wine()
    n_classes = 3
else:
    raise ValueError("Invalid dataset name. Choose 'iris', 'breast_cancer', or 'wine'.")

X, y = data.data, data.target.reshape(-1, 1)

# Standardize features to [0, 1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
n_samples, n_features = X_train.shape

# One-hot encode the target labels for MSE loss
T_onehot = np.eye(n_classes)[y_train.flatten().astype(int)]

# **Step 2: Define Model Hyperparameters**
Time_Limit = 300
hidden_layers = [5, 5]  # Two hidden layers with 5 neurons each
layer_sizes = [n_features] + hidden_layers + [n_classes]  # Output layer has n_classes neurons
z_bounds = [(-1, 1)] * len(hidden_layers)  # Bounds for pre-activations
assert all(lb < 0 and ub > 0 for lb, ub in z_bounds), "z_bounds must include 0"

beta = 0.1    # Layer penalty coefficient
M = 1000      # Big-M constant
alpha = 0.1   # Regularization strength
l1_ratio = 0.5  # Balance between L1 and L2 regularization

# **Step 3: Define the MIP Neural Network Model**
def define_mip_nn(X, T_onehot, layers, bounds, beta=0.1, alpha=0.1, l1_ratio=0.5, M=1000):
    model = pyo.ConcreteModel()
    
    # Define sets
    model.I = range(len(X))  # Training samples
    model.L = range(1, len(layers))  # Layers excluding input
    model.N = {l: range(layers[l]) for l in range(len(layers))}  # Neurons per layer

    # Define variables
    model.a = pyo.Var(
        ((i, l, j) for i in model.I for l in range(len(layers)) for j in model.N[l]),
        domain=pyo.Reals
    )  # Activations
    model.W = pyo.Var(
        ((l, j, k) for l in model.L for j in model.N[l] for k in model.N[l - 1]),
        domain=pyo.Reals
    )  # Weights
    model.b = pyo.Var(
        ((l, j) for l in model.L for j in model.N[l]),
        domain=pyo.Reals
    )  # Biases
    model.z = pyo.Var(
        ((i, l, j) for i in model.I for l in model.L if l < len(layers)-1 for j in model.N[l]),
        domain=pyo.Reals
    )  # Pre-activations
    model.delta = pyo.Var(
        ((i, l, j) for i in model.I for l in model.L if l < len(layers)-1 for j in model.N[l]),
        domain=pyo.Binary
    )  # ReLU indicators
    model.abs_W = pyo.Var(
        ((l, j, k) for l in model.L for j in model.N[l] for k in model.N[l - 1]),
        domain=pyo.NonNegativeReals
    )  # Absolute weights
    model.gamma = pyo.Var(
        (l for l in range(1, len(layers) - 1)),
        domain=pyo.Binary
    )  # Layer presence indicators

    # Constraint: First hidden layer must be active
    model.first_layer_constraint = pyo.Constraint(expr=model.gamma[1] == 1)

    # Input layer initialization
    def input_rule(m, i, j):
        return m.a[i, 0, j] == X[i, j]
    model.input_con = pyo.Constraint(model.I, model.N[0], rule=input_rule)

    # Pre-activation equations
    def pre_activation_rule(m, i, l, j):
        return m.z[i, l, j] == sum(m.W[l, j, k] * m.a[i, l - 1, k] for k in m.N[l - 1]) + m.b[l, j]
    model.pre_act = pyo.Constraint(
        ((i, l, j) for i, l, j in model.z),
        rule=pre_activation_rule
    )

    # ReLU encoding constraints
    model.relu = pyo.ConstraintList()
    for l in range(1, len(layers)-1):
        zmin, zmax = bounds[l-1]
        for i in model.I:
            for j in model.N[l]:
                model.relu.add(model.a[i, l, j] >= 0)
                model.relu.add(model.a[i, l, j] >= model.z[i, l, j])
                model.relu.add(model.a[i, l, j] <= model.z[i, l, j] - zmin * (1 - model.delta[i, l, j]))
                model.relu.add(model.a[i, l, j] <= zmax * model.delta[i, l, j])

    # Output layer (linear)
    def output_rule(m, i, c):
        l = len(layers) - 1
        return m.a[i, l, c] == sum(m.W[l, c, k] * m.a[i, l - 1, k] for k in m.N[l - 1]) + m.b[l, c]
    model.output = pyo.Constraint(model.I, range(n_classes), rule=output_rule)

    # L1-norm linearization
    model.abs_W_constraints = pyo.ConstraintList()
    for l, j, k in model.abs_W:
        model.abs_W_constraints.add(model.abs_W[l, j, k] >= model.W[l, j, k])
        model.abs_W_constraints.add(model.abs_W[l, j, k] >= -model.W[l, j, k])

    # Layer pruning constraints (Big-M)
    model.layer_present = pyo.ConstraintList()
    for l in range(1, len(layers) - 1):
        for j in model.N[l]:
            for k in model.N[l - 1]:
                model.layer_present.add(model.W[l, j, k] <= M * model.gamma[l])
                model.layer_present.add(model.W[l, j, k] >= -M * model.gamma[l])
            model.layer_present.add(model.b[l, j] <= M * model.gamma[l])
            model.layer_present.add(model.b[l, j] >= -M * model.gamma[l])
            for i in model.I:
                model.layer_present.add(model.a[i, l, j] <= M * model.gamma[l])
                model.layer_present.add(model.z[i, l, j] <= M * model.gamma[l])

    # Objective: MSE loss + Regularization
    def objective_rule(m):
        # MSE loss: sum of squared differences between outputs and one-hot targets
        loss = sum(
            (m.a[i, len(layers)-1, c] - T_onehot[i, c])**2
            for i in m.I
            for c in range(n_classes)
        )
        # L1 regularization
        l1_penalty = alpha * l1_ratio * sum(
            m.abs_W[l, j, k] for l, j, k in m.abs_W
        )
        # L2 regularization
        l2_penalty = 0.5 * alpha * (1 - l1_ratio) * sum(
            m.W[l, j, k] ** 2 for l, j, k in m.W
        )
        # Layer sparsity penalty
        layer_penalty = beta * sum(m.gamma[l] for l in m.gamma)
        return loss + l1_penalty + l2_penalty + layer_penalty
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model

# **Step 4: Build and Solve the Model**
model = define_mip_nn(X_train, T_onehot, layer_sizes, z_bounds, beta=beta, alpha=alpha, l1_ratio=l1_ratio, M=M)
solver = pyo.SolverFactory("gurobi")
solver.options.update({
    "TimeLimit": Time_Limit,
    "MIPGap": 0.01,
    "MIPGapAbs": 1e-4,
    "FeasibilityTol": 1e-6,
    "IntFeasTol": 1e-5,
    "OptimalityTol": 1e-6,
    "VarBranch": 1,
    "BranchDir": 0,
    "NodeMethod": 1,
    "NodefileStart": 0.5,
    "Heuristics": 0.5,
    "RINS": 10,
    "PumpPasses": 5,
    "Presolve": 2,
    "CutPasses": 5,
    "Cuts": 2,
    "Threads": 1
})
solver.solve(model, tee=True)

# **Step 5: Evaluate the Model**
# Predict by taking the class with the highest output value
raw_preds = [
    np.argmax([pyo.value(model.a[i, len(layer_sizes)-1, c]) for c in range(n_classes)])
    for i in model.I
]
y_pred = np.array(raw_preds)
cm = confusion_matrix(y_train.flatten(), y_pred)
acc = np.trace(cm) / np.sum(cm)
print(f"\nTraining Accuracy: {acc:.3f}")

# **Step 6: Save Model Parameters**
params = {
    "W": {(l, j, k): pyo.value(model.W[l, j, k]) for l, j, k in model.W},
    "b": {(l, j): pyo.value(model.b[l, j]) for l, j in model.b},
    "gamma": {l: pyo.value(model.gamma[l]) for l in model.gamma},
    "delta": {(i, l, j): pyo.value(model.delta[i, l, j]) for i, l, j in model.delta},
    "alpha": alpha,
    "l1_ratio": l1_ratio,
    "beta": beta
}
with open("trained_nn.pkl", "wb") as f:
    pickle.dump(params, f)

# **Step 7: Sparsity Reporting**
def compute_weight_based_sparsity(params, layer_sizes, threshold=1e-4):
    W = params["W"]
    sparsity_report = {}
    for l in range(1, len(layer_sizes)):
        neuron_sparsity = {}
        for j in range(layer_sizes[l]):
            incoming_weights = [W.get((l, j, k), 0.0) for k in range(layer_sizes[l - 1])]
            is_sparse = all(abs(w) < threshold for w in incoming_weights)
            neuron_sparsity[j] = {
                "nonzero_count": sum(abs(w) >= threshold for w in incoming_weights),
                "sparse": is_sparse
            }
        sparsity_report[l] = neuron_sparsity
    return sparsity_report

sparsity_report = compute_weight_based_sparsity(params, layer_sizes)
for l in sorted(sparsity_report.keys()):
    print(f"\nLayer {l}:")
    for j, stat in sparsity_report[l].items():
        status = "PRUNABLE" if stat["sparse"] else ""
        print(f" - Neuron {j}: {stat['nonzero_count']} active connections {status}")