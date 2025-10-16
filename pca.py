import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

df = pd.read_csv('dados_municipais_consolidados.csv', encoding='latin1')

# Carregar os dados
df_analise = pd.read_csv('dados_municipais_consolidados.csv', encoding='latin1')

# Remover dados nulos
df_analise.dropna(inplace=True)

# Inicializar o LabelEncoder
label_encoder = LabelEncoder()

# Identificar e transformar colunas categóricas
for col in df_analise.select_dtypes(include=['object']).columns:
    df_analise[col] = label_encoder.fit_transform(df_analise[col])

n_amostra = len(df_analise)
print(n_amostra)
df_amostra = df_analise.sample(n=n_amostra, random_state=42)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_amostra)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

df_amostra['PC1'] = pca_result[:, 0]
df_amostra['PC2'] = pca_result[:, 1]

variancia_explicada = pca.explained_variance_ratio_
print("\nVariância explicada por componente:")
for i, v in enumerate(variancia_explicada, start=1):
    print(f"Componente {i}: {v:.2%}")

print(f"\nVariância total explicada pelas 2 primeiras componentes: {sum(variancia_explicada):.2%}")

loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2']
)
print("\nCargas fatoriais (loadings):\n")
print(loadings.round(3))

plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2', data=df_amostra, alpha=0.7)
plt.title('Projeção PCA: IDH + Desempenho Escolar')
plt.xlabel(f'Componente Principal 1 ({variancia_explicada[0]*100:.1f}% var.)')
plt.ylabel(f'Componente Principal 2 ({variancia_explicada[1]*100:.1f}% var.)')
plt.tight_layout()
plt.show()

df_amostra.to_csv('amostra_pca_idh_desempenho.csv', index=False)
loadings.to_csv('pca_loadings_idh_desempenho.csv', index=True)

print("\nArquivos 'amostra_pca_idh_desempenho.csv' e 'pca_loadings_idh_desempenho.csv' salvos com sucesso.")
