import torch

x_int = torch.randint(1, 10, (1,))
print(f"1. Исходный целочисленный тензор: {x_int}, dtype: {x_int.dtype}")

x = x_int.float()
print(f"2. Тензор после преобразования к float32: {x}, dtype: {x.dtype}")

n = 2 
print(f"3. Используется степень n = {n}")

x_grad = x.clone().requires_grad_(True)

x_powered = x_grad ** n
print(f"3.1. x^{n} = {x_powered.item()}")

multiplier = torch.rand(1) * 2 + 1 
x_multiplied = x_powered * multiplier
print(f"3.2. Умножение на {multiplier.item():.2f} = {x_multiplied.item()}")

x_exp = torch.exp(x_multiplied)
print(f"3.3. exp(...) = {x_exp.item()}")


x_exp.backward() 
print(f"4. Производная в точке x={x_grad.item()} равна: {x_grad.grad.item()}")