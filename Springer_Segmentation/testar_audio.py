import joblib
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import os

# (Este script deve estar salvo DENTRO da pasta Springer-Segmentation)

print("--- INICIANDO TESTE COM O MODELO ORIGINAL ---")

# Como o script e o modelo estão na mesma pasta, podemos chamar pelo nome direto
caminho_pkl = r"C:\Users\gusta\TCC\TCC\springer_segmentation_model.pkl"

# 1. COLOQUE AQUI O CAMINHO EXATO DO SEU ÁUDIO DO BMD-HS (Ex: um arquivo da pasta train)
arquivo_wav = r"C:\Users\gusta\TCC\TCC\data\train\AR_016_sit_Aor.wav" 

print(f"1/3 - Carregando modelo treinado: {caminho_pkl}")
modelo_carregado = joblib.load(caminho_pkl)

print(f"2/3 - Lendo áudio do BMD-HS: {os.path.basename(arquivo_wav)}")
rate, audio_bruto = wav.read(arquivo_wav)

# Normalizar o áudio (int16 -> float entre -1 e 1) - Crucial para a matemática funcionar
audio_normalizado = audio_bruto.astype(np.float32)
max_val = np.max(np.abs(audio_normalizado))
if max_val > 0:
    audio_normalizado = audio_normalizado / max_val

print("3/3 - Rodando predição do modelo (Isso fará a extração de wavelets internamente)...")
# O código original é inteligente: ele aceita a onda sonora limpa e extrai as características sozinho!
segmentacao_final = modelo_carregado.predict(audio_normalizado)

print("--- SEGMENTAÇÃO CONCLUÍDA COM SUCESSO! GERANDO GRÁFICO... ---")

# --- VISUALIZAÇÃO DOS RESULTADOS ---
segundos_para_ver = 3

# O relógio do áudio (Rápido: 4000 pontos por segundo)
amostras_audio = int(segundos_para_ver * rate)
tempo_audio = np.linspace(0, segundos_para_ver, amostras_audio)

# O relógio do algoritmo de Springer (Lento: 50 decisões por segundo)
feature_frequency = 50
amostras_seg = int(segundos_para_ver * feature_frequency)
tempo_seg = np.linspace(0, segundos_para_ver, amostras_seg)

plt.figure(figsize=(12, 6))

# Plota a onda sonora real do paciente
plt.plot(tempo_audio, audio_normalizado[:amostras_audio], color='lightblue', label='Sinal do Coração')

# Cria o eixo para a linha de segmentação
ax2 = plt.twinx()
ax2.plot(tempo_seg, segmentacao_final[:amostras_seg], color='red', linewidth=2, label='Fases Segmentadas')

# Formatação do gráfico
ax2.set_yticks([1, 2, 3, 4])
ax2.set_yticklabels(['S1', 'Sístole', 'S2', 'Diástole'])
plt.title('Teste Definitivo de Segmentação - TCC Gustavo')
plt.xlabel('Tempo (segundos)')

# Junta as legendas corrigindo o bug
linhas_1, labels_1 = plt.gca().get_legend_handles_labels()
# Como a onda azul foi o primeiro plot, pegamos ela forçadamente se sumir
if not labels_1:
    labels_1 = ['Sinal do Coração']
    linhas_1 = [plt.Line2D([0], [0], color='lightblue')]

linhas_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(linhas_1 + linhas_2, labels_1 + labels_2, loc='upper right')

plt.tight_layout()
plt.show()