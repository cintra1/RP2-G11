from google.colab import files
import pandas as pd

print("Faça o upload do arquivo do Atlas do Desenvolvimento Humano (IDH):")
uploaded_idh = files.upload()

print("Agora faça o upload do arquivo do Censo Escolar do INEP:")
uploaded_censo = files.upload()

idh_file = list(uploaded_idh.keys())[0]
censo_file = list(uploaded_censo.keys())[0]

df_idh = pd.read_csv(idh_file, encoding='latin1', sep=None, engine='python')
df_censo = pd.read_csv(censo_file, encoding='latin1', sep=None, engine='python')

print("Colunas no dataset do IDH:")
print(df_idh.columns)

print("Colunas no dataset do Censo Escolar:")
print(df_censo.columns)

col_idh = 'id_municipio'
col_censo = 'CO_MUNICIPIO'

df_idh = df_idh.rename(columns={col_idh: 'codigo_municipio'})
df_censo = df_censo.rename(columns={col_censo: 'codigo_municipio'})

df_merged = pd.merge(df_idh, df_censo, on='codigo_municipio', how='inner')  # ou how='left', 'outer', etc.

print("Dataset consolidado:")
print(df_merged.head())

output_filename = 'dataset_consolidado.csv'
df_merged.to_csv(output_filename, index=False)
files.download(output_filename)
