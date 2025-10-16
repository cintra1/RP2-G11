import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# Carregar os dados
df_analise = pd.read_csv('dados_municipais_consolidados.csv', encoding='latin1')

# Remover dados nulos
df_analise.dropna(inplace=True)

# Inicializar o LabelEncoder
label_encoder = LabelEncoder()

# Identificar e transformar colunas categóricas
for col in df_analise.select_dtypes(include=['object']).columns:
    df_analise[col] = label_encoder.fit_transform(df_analise[col])

# Agora, podemos aplicar a normalização
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_analise)

# Definir o número de clusters
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

# Aplicar o K-means
clusters = kmeans.fit_predict(df_scaled)

# Adicionar a coluna de clusters no dataframe original
df_analise['cluster'] = clusters

print(df_analise)
# Mostrar o resumo por cluster (média de cada variável)
print("\nResumo por cluster (médias):\n")
print(df_analise.groupby('cluster').mean(numeric_only=True).round(2))

# Salvar o dataframe com os clusters em um novo arquivo
df_analise.to_csv('resultados_clusters_full.csv', index=False)
print("\nArquivo 'resultados_clusters_full.csv' salvo com sucesso.")

# Gerar gráfico de pairplot com os clusters
sns.scatterplot(data=df_analise, x='expectativa_vida', y='codigo_municipio', hue='cluster')
plt.suptitle('Clusters K-Means: Desempenho Escolar + IDH (Todos os dados)', y=1.02)
plt.tight_layout()
plt.show()


