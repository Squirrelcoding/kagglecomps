import torch
import pandas as pd

from sklearn.model_selection import train_test_split

from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"DEVICE: {device}")

df = pd.read_csv("train.csv")

labels = df["Survived"].to_numpy()

# drop names
df = df.drop(columns=["Survived", "Name", "PassengerId", "Ticket", "Cabin"])

df = pd.get_dummies(df, columns=["Sex", "Pclass", "SibSp", "Parch", "Embarked"], dtype=float)

df = df.fillna(0)

features = df.to_numpy()

labels = torch.from_numpy(labels).type(torch.float)
features = torch.from_numpy(features).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

X_train = X_train.to(device) #pyright: ignore
X_test = X_test.to(device) #pyright: ignore
y_train = y_train.to(device) #pyright: ignore
y_test = y_test.to(device) #pyright: ignore


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

class TitanicModel(nn.Module):
    def __init__(self):
        super().__init__()

        hidden_units = 10

        self.layer_1 = nn.Linear(in_features=24, out_features=hidden_units)
        self.layer_2 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.layer_4 = nn.Linear(in_features=hidden_units, out_features=1)
        self.relu = nn.ReLU() # <- add in ReLU activation function
        self.layers = nn.Sequential(
            nn.Linear(in_features=24, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Sigmoid(),
            nn.Linear(in_features=hidden_units, out_features=1)
        )
        # Can also put sigmoid in the model 
        # This would mean you don't need to use it on the predictions
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Intersperse the ReLU activation function between layers
        return self.layers(x)
model = TitanicModel().to(device)

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.025)

epochs = 7500

for epoch in range(epochs):
    model.train()

    logits = model(X_train).squeeze()
    preds = torch.round(torch.sigmoid(logits))

    loss = loss_fn(logits, y_train)
    acc = accuracy_fn(y_train, preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_preds)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# Save the training columns
training_columns = df.columns  # <--- NEW LINE

# ===== TEST DATA =====
df_app = pd.read_csv("test.csv")
passenger_ids = df_app["PassengerId"].to_numpy()
df_app = df_app.drop(columns=["Name", "PassengerId", "Ticket", "Cabin"])
df_app = pd.get_dummies(df_app, columns=["Sex", "Pclass", "SibSp", "Parch", "Embarked"], dtype=float)

# Align test data columns with training data
df_app = df_app.reindex(columns=training_columns, fill_value=0)  # <--- FIX
df_app = df_app.fillna(0)

features_app = df_app.to_numpy()
features_app = torch.from_numpy(features_app).type(torch.float) #pyright: ignore
features_app = features_app.to(device)

print(features_app[0])
print(X_train[0])
print(f"Shape of tested thing: {features_app[0].shape}")
print(f"Shape of original data: {X_train[0].shape}")

results = []

for id, X in zip(passenger_ids, features_app):
    survived = int(torch.round(torch.sigmoid(model(X).squeeze())))
    results.append((id, survived))

df_out = pd.DataFrame(results, columns=['PassengerId', 'Survived']) #pyright: ignore

df_out.to_csv('titanic_predictions.csv', index=False)


