import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('dados_municipais_consolidados.csv', encoding='latin1')

variaveis_idh = [
    'idhm', 'idhm_e', 'idhm_l', 'idhm_r',
    'renda_pc', 'indice_gini', 'expectativa_vida', 'expectativa_anos_estudo'
]

variaveis_desempenho = [
    'taxa_analfabetismo_15_mais',
    'taxa_freq_liquida_medio',
    'taxa_freq_fundamental_15_17',
    'taxa_atraso_1_fundamental',
    'indice_escolaridade'
]

colunas_analise = variaveis_idh + variaveis_desempenho

df_analise = df[colunas_analise].copy()

df_analise.dropna(inplace=True)

n_amostra = min(1000, len(df_analise))
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
    columns=['PC1', 'PC2'],
    index=colunas_analise
)
print("\nCargas fatoriais (loadings):\n")
print(loadings.round(3))

plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2', data=df_amostra, alpha=0.7)
plt.title('Projeção PCA (amostra de 1000): IDH + Desempenho Escolar')
plt.xlabel(f'Componente Principal 1 ({variancia_explicada[0]*100:.1f}% var.)')
plt.ylabel(f'Componente Principal 2 ({variancia_explicada[1]*100:.1f}% var.)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
plt.title('Cargas Fatoriais (Loadings) das Variáveis nas Duas Primeiras Componentes')
plt.tight_layout()
plt.show()

df_amostra.to_csv('amostra_pca_idh_desempenho.csv', index=False)
loadings.to_csv('pca_loadings_idh_desempenho.csv', index=True)

print("\nArquivos 'amostra_pca_idh_desempenho.csv' e 'pca_loadings_idh_desempenho.csv' salvos com sucesso.")
