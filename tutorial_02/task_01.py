import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# load dataset

iris = load_iris()
X = iris.data
y = iris.target

# split/scale Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# different MLP Configurations
configurations = {
    "1 Hidden Layer - 10 Neurons": (10,),
    "1 Hidden Layer - 20 Neurons": (20,),
    "2 Hidden Layers - 10, 10": (10, 10),
    "2 Hidden Layers - 20, 20": (20, 20),
    "3 Hidden Layers - 10, 10, 10": (10, 10, 10)
}

# train
results = {}
models = {}

for name, layers in configurations.items():

    model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Store results
    results[name] = {
        "accuracy": accuracy,
        "epochs": model.n_iter_,
        "loss": model.loss_
    }

    models[name] = model

print("=" * 70)
print("TASK 1 RESULTS")
print("=" * 70)

for name, result in results.items():

    print("\nConfiguration:", name)
    print("Accuracy:", result["accuracy"])
    print("Epochs:", result["epochs"])
    print("Final Loss:", result["loss"])

best = max(
    results,
    key=lambda x: results[x]["accuracy"]
)

print("\n" + "=" * 70)
print("BEST PERFORMANCE")
print("=" * 70)

print("Configuration:", best)
print("Accuracy:", results[best]["accuracy"])
print("Epochs:", results[best]["epochs"])
print("Final Loss:", results[best]["loss"])

# plot Learning Curves
plt.figure(figsize=(10, 6))

for name, model in models.items():

    plt.plot(
        model.loss_curve_,
        label=name
    )

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.title("Task 1: MLP Learning Curves")

plt.legend()
plt.grid()

plt.show()
