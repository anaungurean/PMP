import matplotlib.pyplot as plt
import pandas as pd

# Încărcarea datelor din fișierul CSV
file_path = 'auto-mpg.csv'
data = pd.read_csv(file_path, sep=',')

# Afișarea primelor câteva rânduri pentru a verifica structura datelor
data.head()
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

# Eliminarea rândurilor cu valori lipsă din 'horsepower'
data_clean = data.dropna(subset=['horsepower'])


# Crearea unui grafic pentru a vizualiza relația dintre 'horsepower' și 'mpg'
plt.figure(figsize=(10, 6))
plt.scatter(data_clean['horsepower'], data_clean['mpg'])
plt.title('Relația dintre CP (Horsepower) și MPG (Miles per Gallon)')
plt.xlabel('CP (Horsepower)')
plt.ylabel('MPG (Miles per Gallon)')
plt.grid(True)
plt.show()