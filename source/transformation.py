import pandas as pd
import numpy as np

print("--- INICIANDO A FUSÃO DE DADOS (PASSO 5) ---")

caminho_salvar = 'dataset_final.csv'

# 1. Carregar as 3 tabelas
print("A carregar os ficheiros CSV...")
df_features = pd.read_csv('features_extraidas.csv')
df_train = pd.read_csv(r'data\train.csv')
df_meta = pd.read_csv(r'data\additional_metadata.csv')

# 2. Reorganizar o Train.csv (Técnica de "Melt")
print("A derreter o train.csv para o formato de áudios individuais...")
colunas_audios = [f'recording_{i}' for i in range(1, 9)]

# Transforma as colunas recording_1...8 em linhas
df_train_linhas = df_train.melt(
    id_vars=['patient_id', 'AS', 'AR', 'MR', 'MS', 'N'], 
    value_vars=colunas_audios, 
    value_name='nome_do_audio_sem_wav'
)

# Remove as linhas vazias (já que alguns pacientes só têm 4 ou 5 áudios, não 8)
df_train_linhas = df_train_linhas.dropna(subset=['nome_do_audio_sem_wav']).copy()

# Adiciona a extensão '.wav' para o nome ficar exatamente igual ao do nosso features.csv
df_train_linhas['arquivo_wav'] = df_train_linhas['nome_do_audio_sem_wav'].astype(str) + '.wav'

# 3. Juntar com os Metadados (Idade, Género, etc.)
print("A anexar os metadados dos pacientes...")
df_rotulos_completos = pd.merge(df_train_linhas, df_meta, on='patient_id', how='left')

# 4. A GRANDE FUSÃO: Cruzar os Rótulos Médicos com a Matemática (Liu et al.)
print("A cruzar os rótulos com as 20 Features de Springer...")
# A coluna no nosso features_liu_et_al.csv chama-se 'patient_id', mas ela guarda o NOME do arquivo.
# Vamos renomear para não haver confusão e o cruzamento ser exato.
df_features = df_features.rename(columns={'patient_id': 'arquivo_wav'})

# O 'inner join' garante que só mantemos os áudios que têm tanto as features extraídas quanto o diagnóstico
dataset_final = pd.merge(df_features, df_rotulos_completos, on='arquivo_wav', how='inner')

# Limpar colunas de lixo (como a 'variable' que dizia se era gravação 1 ou 2)
dataset_final = dataset_final.drop(columns=['variable', 'nome_do_audio_sem_wav'])

# Reorganizar as colunas para ficar elegante (Identificadores -> Diagnóstico -> Metadados -> Features)
colunas_organizadas = ['patient_id', 'arquivo_wav', 'AS', 'AR', 'MR', 'MS', 'N', 
                       'Age', 'Gender', 'Smoker', 'Lives'] + [col for col in df_features.columns if col != 'arquivo_wav']
dataset_final = dataset_final[colunas_organizadas]

""" # 5. Guardar a "Mina de Ouro"
caminho_salvar = 'dataset_final.csv'
dataset_final.to_csv(caminho_salvar, index=False)

print("\n--- SUCESSO ABSOLUTO! ---")
print(f"Total de áudios prontos para a IA: {dataset_final.shape[0]}")
print(f"Total de colunas na tabela final: {dataset_final.shape[1]}")
print(f"Ficheiro guardado como: {caminho_salvar}") """

print("--- PREPARANDO DADOS PARA A INTELIGÊNCIA ARTIFICIAL ---")

""" caminho_dataset = 'dataset_final_machine_learning.csv'
print(f"Lendo o arquivo: {caminho_dataset}") """
df = dataset_final.copy() # Para não mexer no dataset_final original, só para garantir

# 1. Transformar Gênero em Binário (M=1, F=0)
print("Convertendo 'Gender' para binário (M=1, F=0)...")
df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})

# 2. Transformar Moradia em Binário (Urbano=1, Rural=0)
print("Convertendo 'Lives' para binário (U=1, R=0)...")
df['Lives'] = df['Lives'].map({'U': 1, 'R': 0})

# 3. Transformar a Idade (Age) de "30-39" para um número contínuo
# A técnica clássica é pegar o valor do meio. Ex: "30-39" vira 34.5
print("Convertendo 'Age' para números contínuos (pegando o ponto médio)...")
def converter_idade(idade_str):
    if pd.isna(idade_str):
        return np.nan
    try:
        # Pega a string "30-39", separa no traço e calcula a média
        partes = str(idade_str).split('-')
        return (int(partes[0]) + int(partes[1])) / 2.0
    except:
        return np.nan # Se houver alguma idade fora do padrão, deixa vazio

df['Age'] = df['Age'].apply(converter_idade)

# 4. Preencher possíveis valores vazios (NaN) com a média da coluna
# (A Inteligência Artificial odeia buracos vazios na tabela)
print("Preenchendo possíveis dados em branco...")
colunas_para_preencher = ['Age', 'Gender', 'Smoker', 'Lives']
for col in colunas_para_preencher:
    df[col] = df[col].fillna(df[col].mean())

# 5. Salvar o arquivo por cima dele mesmo (agora 100% numérico)
df.to_csv(caminho_salvar, index=False)

print("\n--- TABELA 100% NUMÉRICA E PRONTA! ---")
# Mostra apenas as colunas de metadados para você ver como ficou
print(df[['patient_id', 'Age', 'Gender', 'Smoker', 'Lives']].head())