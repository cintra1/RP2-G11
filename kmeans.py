import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('dataset_consolidado.csv', encoding='latin1')

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

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_analise)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(df_scaled)

df_analise['cluster'] = clusters

sns.pairplot(df_analise, hue='cluster', diag_kind='kde', palette='Set1')
plt.suptitle('Clusters K-Means: Desempenho Escolar + IDH (Todos os dados)', y=1.02)
plt.tight_layout()
plt.show()

print("\nResumo por cluster (médias):\n")
print(df_analise.groupby('cluster').mean(numeric_only=True).round(2))

df_analise.to_csv('resultados_clusters_full.csv', index=False)
print("\nArquivo 'resultados_clusters_full.csv' salvo com sucesso.")
