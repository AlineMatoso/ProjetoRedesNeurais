
# link dataset https://www.kaggle.com/datasets/ayeshasiddiqa123/academic-stress-factors-among-students/data

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Carregar dados
df = pd.read_csv('data_science_student_marks.csv')

# Definir features e target
features = ['sql_marks', 'excel_marks', 'power_bi_marks', 'english_marks', 'age']
X = df[features]
y = df['python_marks']

# Padronização dos dados (recomendado para redes neurais)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Criar DataFrame padronizado
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

print("Estatísticas antes da padronização:")
print(X.describe())
print("\nEstatísticas após padronização:")
print(X_scaled_df.describe())

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42
)

print(f"\nDimensões dos conjuntos:")
print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")