import csv
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchviz import make_dot


# Define the PyTorch model
class TextSentimentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super(TextSentimentModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        # Reshape input to (batch_size, sequence_length, input_dim)
        x = x.unsqueeze(1)  # Add a sequence dimension of size 1
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the output of the last LSTM cell
        x = self.fc(x)
        return x


# Prepare dataset
print("Fetching dataset...")
data = fetch_20newsgroups(subset='all', categories=['rec.sport.baseball', 'sci.med'],
                          remove=('headers', 'footers', 'quotes'))
X, y = data.data, data.target

print("Vectorizing dataset...")
# Vectorize text data
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(X).toarray()
y = np.array(y)

print("Splitting dataset...")
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

print("Converting to PyTorch tensors...")
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print("Initializing model...")
# Hyperparameters
input_dim = X_train_tensor.shape[1]
hidden_dim = 256
output_dim = len(set(y))
n_layers = 4
dropout = 0.3
learning_rate = 0.001
batch_size = 256
num_epochs = 64
grid_size = 250  # Time Complexity -> O(N^2 * num_batches)

# Model, loss, and optimizer
model = TextSentimentModel(input_dim, hidden_dim, output_dim, n_layers, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Preparing dataloader...")
# Dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("Starting training loop...")
# Tracking metrics
train_losses = []
test_losses = []
test_accuracies = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor).item()
        test_preds = torch.argmax(test_outputs, dim=1)
        test_acc = (test_preds == y_test_tensor).float().mean().item()

    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

print("Saving the model and vectorizer...")
# Save the model and vectorizer
model_path = "text_sentiment_model.pth"
vectorizer_path = "tfidf_vectorizer.pkl"
torch.save(model.state_dict(), model_path)
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"Model saved to {model_path}")
print(f"Vectorizer saved to {vectorizer_path}")

# Visualizations and metrics
print("Creating visualizations...")
os.makedirs("visualizations", exist_ok=True)

# 1. Feature Importance
print("Generating feature importance visualization...")
feature_importance = np.mean(X_train, axis=0)
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.title("Top 90 Features")
plt.xlabel("Feature Index")
plt.ylabel("Mean Importance")
plt.savefig("visualizations/Top_90_Features.svg")
plt.close()

# 2. Loss Landscape 3D
model.eval()  # Set model to evaluation mode
param = next(model.parameters())  # Use the first parameter for landscape perturbations
param_flat = param.view(-1)

# Define perturbation directions u and v
u = torch.randn_like(param_flat).view(param.shape).to(param.device)
v = torch.randn_like(param_flat).view(param.shape).to(param.device)

# Normalize perturbations
u = 0.01 * u / torch.norm(u)
v = 0.01 * v / torch.norm(v)

# Create grid
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
loss_values = np.zeros((grid_size, grid_size))

# Iterate through the grid to compute losses
for i, dx in enumerate(x):
    print(f"Computing loss for row {i + 1}/{grid_size}...")
    for j, dy in enumerate(y):
        print(f"    Computing loss for column {j + 1}/{grid_size}...")
        param.data += dx * u + dy * v  # Apply perturbation
        loss = 0

        # Compute loss for all batches in data loader
        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(param.device)
            targets = targets.to(param.device)
            outputs = model(inputs)
            loss += criterion(outputs, targets).item()

        loss_values[i, j] = loss  # Store the loss
        param.data -= dx * u + dy * v  # Revert perturbation

# Create a meshgrid for plotting
X, Y = np.meshgrid(x, y)

# Plot the 3D surface using Plotly
fig = go.Figure(data=[go.Surface(z=loss_values, x=X, y=Y, colorscale="Viridis")])
fig.update_layout(
    title="Loss Landscape (Interactive 3D)",
    scene=dict(
        xaxis_title="Perturbation in u",
        yaxis_title="Perturbation in v",
        zaxis_title="Loss",
    ),
)

# Save as an interactive HTML file
fig.write_html("visualizations/3D_Loss_Plot.html")
print(f"3D loss landscape saved as visualizations/3D_Loss_Plot.html")

# 4. Model Summary
print("Saving model summary...")
with open("visualizations/Model_Summary.txt", "w") as f:
    f.write(str(model))

# 5. Model Visualization
print("Creating model visualization...")
sample_input = torch.rand(1, input_dim)
make_dot(model(sample_input), params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render("visualizations/Model_Visualization", format="png")

# 6. Model State Dictionary
print("Saving model state dictionary...")
with open("visualizations/Model_state_dictionary.txt", "w") as f:
    f.write(str(model.state_dict()))

# 7. Neural Network Nodes Graph (Gephi)
print("Creating neural network graph...")
graph = nx.DiGraph()
for i, (name, param) in enumerate(model.named_parameters()):
    graph.add_node(name, size=param.numel())
    if i > 0:
        graph.add_edge(previous_name, name)
    previous_name = name
nx.write_gexf(graph, "visualizations/Neural_Network_Nodes_Graph.gexf")

# 8. Nodes and Edges CSV
print("Exporting nodes and edges to CSV...")
with open("visualizations/Nodes_and_edges_GEPHI.csv", "w", newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Source", "Target", "Weight"])
    for edge in graph.edges(data=True):
        csvwriter.writerow([edge[0], edge[1], edge[2].get("weight", 1)])

# 9. Vectorizer Features
print("Saving vectorizer features...")
with open("visualizations/Vectorizer_features.txt", "w") as f:
    f.write("\n".join(vectorizer.get_feature_names_out()))

# 10. Visualize Activation
print("Visualizing activations for sample input...")
sample_output = model(X_train_tensor[:1])
plt.figure(figsize=(10, 6))
plt.bar(range(sample_output.size(1)), sample_output.detach().numpy()[0])
plt.title("Activations for Sample Input")
plt.xlabel("Neuron Index")
plt.ylabel("Activation Value")
plt.savefig("visualizations/Visualize_Activation.png")
plt.close()

# 11. Visualize t-SNE
# Apply t-SNE on the training data
features_embedded = TSNE(n_components=2).fit_transform(X_train)

# Ensure y_train has the same number of elements as features_embedded
assert len(y_train) == len(features_embedded)

# Plot the t-SNE result with training labels
plt.figure(figsize=(10, 6))
plt.scatter(features_embedded[:, 0], features_embedded[:, 1], c=y_train, cmap='viridis', alpha=0.7)
plt.title("t-SNE Visualization")
plt.savefig("visualizations/Visualize_t-SNE.png")
plt.close()

# 12. Weight Distribution
print("Creating weight distribution plot...")
all_weights = torch.cat([param.view(-1) for param in model.parameters()])
sns.histplot(all_weights.detach().numpy(), kde=True, bins=50)
plt.title("Weight Distribution")
plt.savefig("visualizations/Weight_Distribution.png")
plt.close()
