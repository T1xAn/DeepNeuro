import numpy as np


random_array = np.random.randint(1, 101, size=10)


even_numbers = np.where(random_array % 2 == 0, random_array, 0)
even_sum = np.sum(even_numbers)

print(f"Сгенерированный массив: {random_array}")
print(f"Сумма четных чисел: {even_sum}")
