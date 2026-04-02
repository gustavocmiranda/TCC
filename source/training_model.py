import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("--- INICIANDO O TREINAMENTO DA INTELIGÊNCIA ARTIFICIAL ---")

# 1. Carregar a nossa matriz limpa
caminho_dataset = 'dataset_final.csv'
df = pd.read_csv(caminho_dataset)

df = df.dropna(how='any') # Remove linhas com dados faltantes, se houver

# 2. Separar o X (Variáveis Preditoras) e o y (O que queremos adivinhar)
# Vamos retirar os IDs e TODOS os rótulos de doença do X
colunas_para_remover = ['patient_id', 'arquivo_wav', 'AS', 'AR', 'MR', 'MS', 'N']
X = df.drop(columns=colunas_para_remover)

# O nosso Alvo (Target) neste primeiro teste será a Estenose Aórtica (AS)
y = df['AS'] 

# 3. Dividir os dados em Treino e Teste
# 80% dos áudios serão usados para a IA estudar.
# 20% ficarão escondidos para aplicarmos a "prova final" e ver se ela aprendeu mesmo.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Áudios para Treino: {X_train.shape[0]}")
print(f"Áudios para Teste (Prova): {X_test.shape[0]}")

# 4. Criar e Treinar o "Cérebro" da Floresta Aleatória
print("\nTreinando o Random Forest (isso é muito rápido)...")
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train) # É aqui que a máquina aprende!

# 5. A Prova Final: Fazer as previsões nos 20% escondidos
previsoes = modelo_rf.predict(X_test)

# 6. Avaliar o Boletim de Notas
acuracia = accuracy_score(y_test, previsoes)
print(f"\n--- RESULTADOS DA PROVA ---")
print(f"Acurácia Geral: {acuracia * 100:.2f}%\n")

print("Relatório de Classificação (Precision / Recall / F1-Score):")
print(classification_report(y_test, previsoes))

print("Matriz de Confusão:")
print("[[Verdadeiros Negativos , Falsos Positivos]")
print(" [Falsos Negativos    , Verdadeiros Positivos]]")
print(confusion_matrix(y_test, previsoes))

# 7. O Bônus do TCC: O que a máquina achou mais importante?
print("\n--- TOP 5 VARIÁVEIS MAIS IMPORTANTES PARA DETECTAR ESTENOSE AÓRTICA ---")
importancias = modelo_rf.feature_importances_
# Cria uma tabelinha com os nomes das colunas e a importância de cada uma
features_importantes = pd.DataFrame({'Feature': X.columns, 'Importancia': importancias})
features_importantes = features_importantes.sort_values(by='Importancia', ascending=False)
print(features_importantes.head(5).to_string(index=False))