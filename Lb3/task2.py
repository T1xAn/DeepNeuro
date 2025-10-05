import pandas as pd
import torch
import torch.nn as nn

df = pd.read_csv('data.csv', header=None)

X = df.iloc[:, :4].values
y = df.iloc[:, 4].values

unique_labels = list(set(y))
label_to_num = {label: i for i, label in enumerate(unique_labels)}
y_numeric = [label_to_num[label] for label in y]

print(f"Найдены классы: {unique_labels}")
print(f"Соответствие: {label_to_num}")

# Преобразование в тензоры PyTorch
X = torch.FloatTensor(X)
y = torch.LongTensor(y_numeric)

# Простая нейронная сеть
model = nn.Sequential(
    nn.Linear(4, 10),
    nn.ReLU(),
    nn.Linear(10, 3)
)

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Обучение на всех данных
for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Проверка точности
with torch.no_grad():
    _, predicted = torch.max(model(X), 1)
    accuracy = (predicted == y).float().mean()
    print(f'\nТочность: {accuracy * 100:.2f}%')