import sys, numpy as np, scipy, pandas as pd, sklearn, seaborn as sns
print("--- TEST DE ENTORNO OPTIMIZADO ---")
print(f"Python: {sys.version.split()[0]}")
print(f"NumPy:  {np.__version__}")
print(f"SciPy:  {scipy.__version__}")
print(f"Pandas: {pd.__version__}")

# Prueba de potencia bruta
print("Calculando matriz 2000x2000...")
import time
a = np.random.randn(2000, 2000)
start = time.time()
np.dot(a, a)
print(f"¡Éxito! Tiempo: {time.time() - start:.4f} segundos.")
