# %%
try:
    import cupy as cp

    # Test if CuPy can actually access CUDA and random number generator
    cp.cuda.Device(0).compute_capability
    cp.random.seed(1)  # Test if random number generator works
    print("Using CuPy (GPU acceleration)")
except (ImportError, Exception) as e:
    import numpy as cp

    print(f"CuPy not available or CUDA error ({type(e).__name__}), using NumPy (CPU)")
from sklearn.datasets import fetch_openml
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


class CrossEntropy:
    def __init__(self):
        self.softmax_output = None
        self.target = None
        self.batch_size = None

    def forward(self, logits, target):
        # Handle both single samples and batches
        if logits.ndim == 1:
            # Single sample case - reshape to batch of size 1
            logits = logits.reshape(1, -1)
            target = target.reshape(1, -1)

        # Apply softmax per sample (along axis=1)
        # Subtract max for numerical stability, then exponentiate
        exp_logits = cp.exp(logits - cp.max(logits, axis=1, keepdims=True))
        # Divide by sum of exponentiated logits per sample (along axis=1)
        self.softmax_output = exp_logits / cp.sum(exp_logits, axis=1, keepdims=True)

        self.target = target
        self.batch_size = logits.shape[0]  # Store batch size

        # Compute cross entropy loss per sample, then average over the batch
        # Use a small epsilon for numerical stability with log(0)
        log_softmax = cp.log(self.softmax_output + 1e-15)
        # Only consider the log-probabilities of the true classes
        loss_per_sample = -cp.sum(
            target * log_softmax, axis=1
        )  # Sum over classes for each sample

        # Return the average loss over the batch
        return cp.mean(loss_per_sample)

    def backward(self):
        grad = (self.softmax_output - self.target) / self.batch_size
        return grad

    def __call__(self, logits, target):
        return self.forward(logits, target)


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return cp.maximum(0, x)

    def backward(self, grad):
        return cp.where(self.input > 0, grad, 0)

    def __call__(self, x):
        return self.forward(x)


class LeakyReLU:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.input = None

    def forward(self, x):
        self.input = x
        return cp.where(x > 0, x, self.alpha * x)

    def backward(self, grad):
        return cp.where(self.input > 0, grad, self.alpha * grad)

    def __call__(self, x):
        return self.forward(x)


class Sigmoid:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return 1 / (1 + cp.exp(-x))

    def backward(self, grad):
        return grad * (1 - self.input) * self.input

    def __call__(self, x):
        return self.forward(x)


class Adam:
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # Global time step, increments once per batch

        # Initialize moment estimates based on layer type
        self.m = []
        self.v = []
        for layer in self.params:
            if hasattr(layer, "dendrite_W"):  # DendriticLayer
                self.m.append(
                    [
                        cp.zeros_like(layer.dendrite_W),
                        cp.zeros_like(layer.dendrite_b),
                        cp.zeros_like(layer.soma_W),
                        cp.zeros_like(layer.soma_b),
                    ]
                )
                self.v.append(
                    [
                        cp.zeros_like(layer.dendrite_W),
                        cp.zeros_like(layer.dendrite_b),
                        cp.zeros_like(layer.soma_W),
                        cp.zeros_like(layer.soma_b),
                    ]
                )
            else:  # LinearLayer
                self.m.append([cp.zeros_like(layer.W), cp.zeros_like(layer.b)])
                self.v.append([cp.zeros_like(layer.W), cp.zeros_like(layer.b)])

    def zero_grad(self):
        for layer in self.params:
            if hasattr(layer, "dendrite_W"):  # DendriticLayer
                layer.dendrite_dW = 0.0
                layer.dendrite_db = 0.0
                layer.soma_dW = 0.0
                layer.soma_db = 0.0
            else:  # LinearLayer
                layer.dW = 0.0
                layer.db = 0.0

    def step(self):
        self.t += 1  # Increment global time step
        for i, layer in enumerate(self.params):
            if hasattr(layer, "dendrite_W"):  # DendriticLayer
                grads = [
                    layer.dendrite_dW,
                    layer.dendrite_db,
                    layer.soma_dW,
                    layer.soma_db,
                ]
                params = [
                    layer.dendrite_W,
                    layer.dendrite_b,
                    layer.soma_W,
                    layer.soma_b,
                ]
            else:  # LinearLayer
                grads = [layer.dW, layer.db]
                params = [layer.W, layer.b]

            # Update moment estimates and parameters for each parameter
            for j, (grad, param) in enumerate(zip(grads, params)):
                # Update first moment estimate
                self.m[i][j] = self.beta1 * self.m[i][j] + (1 - self.beta1) * grad
                # Update second moment estimate
                self.v[i][j] = self.beta2 * self.v[i][j] + (1 - self.beta2) * (grad**2)

                # Bias correction
                m_hat = self.m[i][j] / (1 - self.beta1**self.t)
                v_hat = self.v[i][j] / (1 - self.beta2**self.t)

                # Update parameters
                param -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)

    def __call__(self):
        return self.step()


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self):
        """Return a list of layers that have a Weight vectors"""
        params = []
        for layer in self.layers:
            if hasattr(layer, "W") or hasattr(layer, "soma_W"):
                params.append(layer)
        return params

    def num_params(self):
        num_params = 0
        for layer in self.layers:
            if hasattr(layer, "num_params"):
                num_params += layer.num_params()
        return num_params

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LinearLayer:
    """A fully connected, feed forward layer"""

    def __init__(self, in_dim, out_dim):
        self.W = cp.random.randn(out_dim, in_dim) * cp.sqrt(
            2.0 / (in_dim)
        )  # He init, for ReLU
        self.b = cp.zeros(out_dim)
        self.dW = 0.0
        self.db = 0.0
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, grad):
        self.dW = grad.T @ self.x
        self.db = grad.sum(axis=0)
        grad = grad @ self.W
        return grad

    def num_params(self):
        return self.W.size + self.b.size

    def __call__(self, x):
        return self.forward(x)


class DendriticLayer:
    """A sparse dendritic layer, consiting of dendrites and somas"""

    def __init__(
        self,
        in_dim,
        n_neurons,
        n_dendrite_inputs,
        n_dendrites,
    ):
        self.in_dim = in_dim
        self.n_dendrite_inputs = n_dendrite_inputs
        self.n_neurons = n_neurons
        self.n_dendrites = n_dendrites
        self.n_soma_connections = n_dendrites * n_neurons
        self.n_synaptic_connections = n_dendrite_inputs * n_dendrites * n_neurons

        self.dendrite_W = cp.random.randn(self.n_soma_connections, in_dim) * cp.sqrt(
            2.0 / (in_dim)
        )  # He init, for ReLU
        self.dendrite_b = cp.zeros((self.n_soma_connections))
        self.dendrite_dW = 0.0
        self.dendrite_db = 0.0
        self.dendrite_activation = LeakyReLU()

        self.soma_W = cp.random.randn(n_neurons, self.n_soma_connections) * cp.sqrt(
            2.0 / (self.n_soma_connections)
        )  # He init, for ReLU
        self.soma_b = cp.zeros(n_neurons)
        self.soma_dW = 0.0
        self.soma_db = 0.0
        self.soma_activation = LeakyReLU()

        # inputs to save for backprop
        self.dendrite_x = None
        self.soma_x = None

        # sample soma mask:
        # [[1, 1, 0, 0]
        #  [0, 0, 1, 1]]
        # number of 1 per row is n_dendrites, rest 0. every column only has 1 entry
        # number of rows equals n_neurons, number of columns eqais n_soma_connections
        # it is a step pattern, so the first n_dendrites entries of the first row are one.
        self.soma_mask = cp.zeros((n_neurons, self.n_soma_connections))
        for i in range(n_neurons):
            start_idx = i * n_dendrites
            end_idx = start_idx + n_dendrites
            self.soma_mask[i, start_idx:end_idx] = 1

        # mask out unneeded weights, thus making weights sparse
        self.soma_W = self.soma_W * self.soma_mask

        # sample dendrite mask
        # for each dendrite sample n_dendrite_inputs from the input array
        self.dendrite_mask = cp.zeros((self.n_soma_connections, in_dim))
        for i in range(self.n_soma_connections):
            # sample without replacement from possible input for a given dendrite from the whole input
            input_idx = cp.random.choice(
                cp.arange(in_dim), size=n_dendrite_inputs, replace=False
            )
            self.dendrite_mask[i, input_idx] = 1

        # mask out unneeded weights, thus making weights sparse
        self.dendrite_W = self.dendrite_W * self.dendrite_mask

    def forward(self, x):
        # dendrites forward pass
        self.dendrite_x = x
        x = x @ self.dendrite_W.T + self.dendrite_b
        x = self.dendrite_activation(x)

        # soma forward pass
        self.soma_x = x
        x = x @ self.soma_W.T + self.soma_b
        x = self.soma_activation(x)
        return x

    def backward(self, grad):
        grad = self.soma_activation.backward(grad)

        # soma back pass, multiply with mask to keep only valid gradients
        self.soma_dW = grad.T @ self.soma_x * self.soma_mask
        self.soma_db = grad.sum(axis=0)
        soma_grad = grad @ self.soma_W

        soma_grad = self.dendrite_activation.backward(soma_grad)

        # dendrite back pass
        self.dendrite_dW = soma_grad.T @ self.dendrite_x * self.dendrite_mask
        self.dendrite_db = soma_grad.sum(axis=0)
        dendrite_grad = soma_grad @ self.dendrite_W

        return dendrite_grad

    def num_params(self):
        print(
            f"\nparameters: dendrite_mask: {cp.sum(self.dendrite_mask)}, dendrite_b: {self.dendrite_b.size}, soma_W: {cp.sum(self.soma_mask)}, soma_b: {self.soma_b.size}"
        )
        return int(
            cp.sum(self.dendrite_mask)
            + self.dendrite_b.size
            + cp.sum(self.soma_mask)
            + self.soma_b.size
        )

    def __call__(self, x):
        return self.forward(x)


def load_mnist_data(
    dataset="mnist",
    subset_size=None,
):
    """
    Download and load the MNIST or Fashion-MNIST dataset.

    Args:
        dataset (str): Dataset to load - either "mnist" or "fashion-mnist"
        normalize (bool): If True, normalize pixel values to [0, 1]
        flatten (bool): If True, flatten 28x28 images to 784-dimensional vectors
        one_hot (bool): If True, convert labels to one-hot encoding
        subset_size (int): If specified, return only a subset of the data

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
            X_train, X_test: Input features
            y_train, y_test: Target labels
    """
    # Map dataset names to OpenML dataset identifiers
    dataset_mapping = {"mnist": "mnist_784", "fashion-mnist": "Fashion-MNIST"}

    if dataset not in dataset_mapping:
        raise ValueError(
            f"Dataset must be one of {list(dataset_mapping.keys())}, got '{dataset}'"
        )

    dataset_name = dataset_mapping[dataset]
    print(f"Loading {dataset.upper()} dataset...")

    # Download dataset
    data = fetch_openml(
        dataset_name, version=1, as_frame=False, parser="auto", cache=True
    )

    X, y = data.data, data.target.astype(int)

    # Split into train and test (last 10k samples for test, rest for train)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Normalize pixel values and convert to GPU arrays
    # Convert to float32 first
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Calculate global mean and std from training data
    mean_val = X_train.mean()
    std_val = X_train.std()

    # Standardize to mean=0, std=1
    X_train = (X_train - mean_val) / std_val
    X_test = (X_test - mean_val) / std_val

    # Convert to CuPy arrays
    X_train = cp.array(X_train)
    X_test = cp.array(X_test)

    # Convert labels to one-hot encoding
    def to_one_hot(labels, n_classes=10):
        one_hot_labels = cp.zeros((len(labels), n_classes))
        one_hot_labels[cp.arange(len(labels)), labels] = 1
        return one_hot_labels

    y_train = to_one_hot(cp.array(y_train))
    y_test = to_one_hot(cp.array(y_test))

    # Use subset if specified
    if subset_size is not None:
        X_train, y_train = X_train[:subset_size], y_train[:subset_size]
        X_test, y_test = (
            X_test[: subset_size // 6],
            y_test[: subset_size // 6],
        )  # Keep proportional test size

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")

    return X_train, y_train, X_test, y_test


def create_batches(X, y, batch_size=128, shuffle=True, drop_last=False):
    n_samples = len(X)
    # shuffle data
    if shuffle:
        indices = cp.arange(n_samples)
        cp.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    for i in range(0, n_samples, batch_size):
        if drop_last and i + batch_size > n_samples:
            break
        X_batch = X[i : i + batch_size]
        y_batch = y[i : i + batch_size]
        yield X_batch, y_batch


def train(
    X_train,
    y_train,
    X_test,
    y_test,
    model,
    criterion,
    optimiser,
    n_epochs=2,
    batch_size=128,
):
    train_losses = []
    accuracy = []
    test_losses = []
    test_accuracy = []
    n_samples = len(X_train)
    num_batches_per_epoch = (n_samples + batch_size - 1) // batch_size
    total_batches = n_epochs * num_batches_per_epoch

    with tqdm(total=total_batches, desc="Training ") as pbar:
        for epoch in range(n_epochs):
            train_loss = 0.0
            correct_pred = 0.0
            for batch_idx, (X, target) in enumerate(
                create_batches(
                    X_train, y_train, batch_size, shuffle=True, drop_last=True
                )
            ):
                # forward pass
                pred = model(X)
                batch_loss = criterion(pred, target)
                train_loss += batch_loss
                # if most likely prediction equals target add to correct predictions
                batch_correct = cp.sum(
                    cp.argmax(pred, axis=1) == cp.argmax(target, axis=1)
                )
                correct_pred += batch_correct

                # backward pass
                optimiser.zero_grad()
                grad = criterion.backward()
                model.backward(grad)
                optimiser.step()

                # Update progress bar
                pbar.set_postfix(
                    {
                        "Epoch": f"{epoch + 1}/{n_epochs}",
                        "Batch": f"{batch_idx + 1}/{num_batches_per_epoch}",
                        "Loss": f"{float(batch_loss):.4f}",
                    }
                )
                pbar.update(1)
            # evaluate on test set
            epoch_test_loss, epoch_test_accuracy = evaluate(
                X_test, y_test, model, criterion
            )
            normalised_train_loss = train_loss / num_batches_per_epoch
            train_losses.append(float(normalised_train_loss))
            epoch_accuracy = correct_pred / n_samples
            accuracy.append(float(epoch_accuracy))
            test_losses.append(float(epoch_test_loss))
            test_accuracy.append(float(epoch_test_accuracy))
    return train_losses, accuracy, test_losses, test_accuracy


def evaluate(
    X_test,
    y_test,
    model,
    criterion,
    batch_size=1024,  # higher batch size for testing
):
    n_samples = len(X_test)
    test_loss = 0.0
    correct_pred = 0.0
    num_batches_per_epoch = (n_samples + batch_size - 1) // batch_size
    for X, target in create_batches(X_test, y_test, batch_size, shuffle=False):
        # forward pass
        pred = model(X)
        batch_loss = criterion(pred, target)
        test_loss += batch_loss
        # if most likely prediction equals target add to correct predictions
        batch_correct = cp.sum(cp.argmax(pred, axis=1) == cp.argmax(target, axis=1))
        correct_pred += batch_correct
    normalised_test_loss = test_loss / num_batches_per_epoch
    accuracy = correct_pred / n_samples
    return float(normalised_test_loss), float(accuracy)


# %%

# for repoducability
cp.random.seed(1287305233)

# data config
dataset = "mnist"  # "mnist", "fashion-mnist"

# config
n_epochs = 5  # 15 MNIST, 20 Fashion-MNIST
lr_1 = 0.001  # 0.002
lr_2 = 0.001  # 0.002
batch_size = 32

in_dim = 28 * 28  # Image dimensions (28x28 MNIST)
n_classes = 10

# dendritic model config
n_dendrite_inputs = 8  # 32 / 16
n_dendrites = 4  # 23 / 31
n_neurons = 10  # 10 / 14

# vanilla model config
hidden_dim = 10  # 10


# Sparse Dendritic Model, 3 layers
model_1 = Sequential(
    [
        DendriticLayer(
            in_dim,
            n_neurons,
            n_dendrite_inputs=n_dendrite_inputs,
            n_dendrites=n_dendrites,
        ),
        # LeakyReLU(),
        # LinearLayer(n_neurons, n_classes),
    ]
)

# Base MLP model, 3 layers
model_2 = Sequential(
    [
        LinearLayer(in_dim, hidden_dim),
        LeakyReLU(),
        LinearLayer(hidden_dim, hidden_dim),
        LeakyReLU(),
        LinearLayer(hidden_dim, n_classes),
    ]
)
optimiser_1 = Adam(model_1.params(), lr=lr_1)
optimiser_2 = Adam(model_2.params(), lr=lr_2)

print(f"number of model_1 params: {model_1.num_params()}")
print(f"number of model_2 params: {model_2.num_params()}")

config = [
    {
        "name": "Sparse Dendritic",
        "model": model_1,
        "optimiser": optimiser_1,
        "lr": lr_1,
    },
    #     {
    #         "name": "Base MLP",
    #         "model": model_2,
    #         "optimiser": optimiser_2,
    #         "lr": lr_2,
    #     },
]

criterion = CrossEntropy()

# load data
X_train, y_train, X_test, y_test = load_mnist_data(dataset=dataset)

# Define colors for plotting
colors = [
    "green",
    "blue",
    "red",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]

# Train all models in config
results = []
for i, model_config in enumerate(config):
    model_name = model_config["name"]
    model = model_config["model"]
    optimiser = model_config["optimiser"]

    print(f"Training {model_name} model...")
    train_losses, train_accuracy, test_losses, test_accuracy = train(
        X_train,
        y_train,
        X_test,
        y_test,
        model,
        criterion,
        optimiser,
        n_epochs,
        batch_size,
    )

    # Store results
    results.append(
        {
            "name": model_name,
            "train_losses": train_losses,
            "train_accuracy": train_accuracy,
            "test_losses": test_losses,
            "test_accuracy": test_accuracy,
            "color": colors[i % len(colors)],
        }
    )

    print(f"Train accuracy {model_name}: {round(train_accuracy[-1] * 100, 1)}%")
    print(f"Test accuracy {model_name}: {round(test_accuracy[-1] * 100, 1)}%")
    print(f"Final train loss {model_name}: {round(train_losses[-1], 4)}")
    print(f"Final test loss {model_name}: {round(test_losses[-1], 4)}")
    print("-" * 50)

# Plot accuracy comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for result in results:
    plt.plot(
        result["train_accuracy"],
        label=f"{result['name']} Train",
        color=result["color"],
        linestyle="--",
    )
    plt.plot(
        result["test_accuracy"], label=f"{result['name']} Test", color=result["color"]
    )
plt.title("Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot loss comparison
plt.subplot(1, 2, 2)
for result in results:
    plt.plot(
        result["train_losses"],
        label=f"{result['name']} Train",
        color=result["color"],
        linestyle="--",
    )
    plt.plot(
        result["test_losses"], label=f"{result['name']} Test", color=result["color"]
    )
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final comparison summary
print("\n" + "=" * 60)
print("FINAL RESULTS COMPARISON")
print("=" * 60)
for result in results:
    print(
        f"{result['name']:20} | Train Acc: {result['train_accuracy'][-1] * 100:5.1f}% | "
        f"Test Acc: {result['test_accuracy'][-1] * 100:5.1f}% | "
        f"Train Loss: {result['train_losses'][-1]:6.4f} | "
        f"Test Loss: {result['test_losses'][-1]:6.4f}"
    )
print("=" * 60)
