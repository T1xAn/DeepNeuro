import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


data = pd.read_csv('dataset_simple.csv')
print(f"Размер датасета: {data.shape}")

X = data.iloc[:, :2].values
y = data.iloc[:, 2].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_dataset = TensorDataset(
    torch.FloatTensor(X_train_scaled), 
    torch.LongTensor(y_train)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test_scaled), 
    torch.LongTensor(y_test)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"\nРазмер обучающей выборки: {len(train_dataset)}")
print(f"Размер тестовой выборки: {len(test_dataset)}")

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x

input_size = 2
hidden_size = 64
output_size = 2 

model = NeuralNetwork(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs=100):
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_features, batch_labels in train_loader:

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'"Эпоха" [{epoch+1}/{epochs}], Потери: {avg_loss:.4f}')
    
    return train_losses

print("\nНачало обучения")
train_losses = train_model(model, train_loader, criterion, optimizer, epochs=100)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'\nТочность на тестовой выборке: {accuracy:.2f}%')
    return accuracy

accuracy = evaluate_model(model, test_loader)

def predict_customer(age, income, model, scaler):
    model.eval()
    customer_data = np.array([[age, income]])
    customer_scaled = scaler.transform(customer_data)
    
    with torch.no_grad():
        customer_tensor = torch.FloatTensor(customer_scaled)
        output = model(customer_tensor)
        probability, prediction = torch.max(output, 1)
    
    result = "купит" if prediction.item() == 1 else "не купит"
    confidence = probability.item()
    
    print(f"\nПредсказание для покупателя (возраст: {age}, доход: {income}):")
    print(f"Вероятность: {confidence:.2%}")
    print(f"Результат: {result}")
    
    return result, confidence

print("\nПроверка предсказания")

test_customers = [
    (25, 30000),
    (45, 75000),
    (35, 50000),
    (60, 90000)
]

for age, income in test_customers:
    predict_customer(age, income, model, scaler)
    print("-" * 30)