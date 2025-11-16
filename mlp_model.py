import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configurações para reproduibilidade
np.random.seed(42)
tf.random.set_seed(42)

# 1. CARREGAR E EXPLORAR OS DADOS
df = pd.read_csv('student_lifestyle_dataset.csv')
print("📊 Dimensões do dataset:", df.shape)
print("\n🔍 Primeiras linhas:")
print(df.head())
print("\n📈 Estatísticas descritivas:")
print(df.describe())
print("\n🎯 Distribuição da variável target:")
print(df['Stress_Level'].value_counts())

# 2. PRÉ-PROCESSAMENTO
# Selecionar features e target
features = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 
           'Physical_Activity_Hours_Per_Day', 'GPA']
X = df[features]
y = df['Stress_Level']

# Codificar o target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n🔢 Classes codificadas: {le.classes_} -> {le.transform(le.classes_)}")

# Normalizar as features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n📊 Divisão dos dados:")
print(f"Treino: {X_train.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")

# 3. CONSTRUIR O MODELO MLP
def create_mlp_model(activation_function='relu', learning_rate=0.001):
    model = keras.Sequential([
        layers.Dense(64, activation=activation_function, input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation=activation_function),
        layers.Dense(16, activation=activation_function),
        layers.Dense(3, activation='softmax')  # 3 classes: Low, Moderate, High
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 4. TREINAR COM DIFERENTES FUNÇÕES DE ATIVAÇÃO
activation_functions = ['relu', 'tanh']
history_dict = {}
models = {}

print("\n🧪 TREINANDO COM DIFERENTES FUNÇÕES DE ATIVAÇÃO...")

for activation in activation_functions:
    print(f"\n📍 Função de ativação: {activation.upper()}")
    
    # Criar e compilar modelo
    model = create_mlp_model(activation_function=activation, learning_rate=0.001)
    models[activation] = model
    
    # Treinar o modelo
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    history_dict[activation] = history
    
    # Avaliar no conjunto de teste
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Acurácia no teste: {test_accuracy:.4f}")

# 5. AVALIAÇÃO E COMPARAÇÃO
print("\n" + "="*50)
print("📊 COMPARAÇÃO FINAL DOS MODELOS")
print("="*50)

best_accuracy = 0
best_activation = None

for activation in activation_functions:
    model = models[activation]
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 {activation.upper()}:")
    print(f"   Acurácia: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_activation = activation

print(f"\n🏆 MELHOR MODELO: {best_activation.upper()} (Acurácia: {best_accuracy:.4f})")

# 6. GRÁFICOS DE DESEMPENHO
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Gráfico de acurácia
for activation in activation_functions:
    history = history_dict[activation]
    axes[0,0].plot(history.history['accuracy'], label=f'{activation} - Treino')
    axes[0,0].plot(history.history['val_accuracy'], label=f'{activation} - Validação', linestyle='--')
axes[0,0].set_title('Acurácia durante o Treinamento')
axes[0,0].set_xlabel('Época')
axes[0,0].set_ylabel('Acurácia')
axes[0,0].legend()
axes[0,0].grid(True)

# Gráfico de perda
for activation in activation_functions:
    history = history_dict[activation]
    axes[0,1].plot(history.history['loss'], label=f'{activation} - Treino')
    axes[0,1].plot(history.history['val_loss'], label=f'{activation} - Validação', linestyle='--')
axes[0,1].set_title('Perda durante o Treinamento')
axes[0,1].set_xlabel('Época')
axes[0,1].set_ylabel('Perda')
axes[0,1].legend()
axes[0,1].grid(True)

# Matriz de confusão para o melhor modelo
best_model = models[best_activation]
y_pred_proba = best_model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1,0])
axes[1,0].set_title(f'Matriz de Confusão - {best_activation.upper()}')
axes[1,0].set_xlabel('Predito')
axes[1,0].set_ylabel('Real')

# Gráfico de comparação de acurácia final
final_accuracies = [accuracy_score(y_test, np.argmax(models[act].predict(X_test), axis=1)) 
                   for act in activation_functions]
bars = axes[1,1].bar(activation_functions, final_accuracies, color=['skyblue', 'lightcoral'])
axes[1,1].set_title('Acurácia Final por Função de Ativação')
axes[1,1].set_ylabel('Acurácia')
axes[1,1].set_ylim(0, 1)

# Adicionar valores nas barras
for bar, acc in zip(bars, final_accuracies):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                  f'{acc:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('desempenho_modelos.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. RELATÓRIO DE CLASSIFICAÇÃO DETALHADO
print("\n" + "="*50)
print("📋 RELATÓRIO DE CLASSIFICAÇÃO - MELHOR MODELO")
print("="*50)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 8. SALVAR O MELHOR MODELO
best_model.save('melhor_modelo_stress.h5')
print(f"\n💾 Melhor modelo salvo como: 'melhor_modelo_stress.h5'")

# 9. EXEMPLO DE PREDIÇÃO
print("\n🎯 EXEMPLO DE PREDIÇÃO:")
sample_idx = 0
sample = X_test[sample_idx].reshape(1, -1)
prediction = best_model.predict(sample)
predicted_class = le.classes_[np.argmax(prediction)]

print(f"Amostra {sample_idx}:")
print(f"  Features: {scaler.inverse_transform(sample)[0]}")
print(f"  Classe real: {le.classes_[y_test[sample_idx]]}")
print(f"  Classe predita: {predicted_class}")
print(f"  Probabilidades: {dict(zip(le.classes_, prediction[0]))}")