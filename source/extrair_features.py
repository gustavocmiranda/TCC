import os
import sys
import joblib
import scipy.io.wavfile as wav
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 1. A CIRURGIA DO CAMINHO (RESOLVE O ERRO DE MÓDULO) ---
# Caminho absoluto da sua pasta do TCC
caminho_tcc = r"C:\Users\gusta\TCC\TCC"
pasta_repositorio = os.path.join(caminho_tcc, "Springer_Segmentation")
 
# Força o Python a ler a pasta do Springer ANTES de carregar o modelo
sys.path.insert(0, pasta_repositorio)

print("--- INICIANDO EXTRAÇÃO DE 20 FEATURES (LIU ET AL.) ---")

# --- 2. CONFIGURAÇÃO DOS FICHEIROS ---
# Procura o modelo (vê se está na raiz ou dentro da pasta do Springer)
caminho_pkl = os.path.join(pasta_repositorio, "springer_segmentation_model.pkl")
if not os.path.exists(caminho_pkl):
    caminho_pkl = os.path.join(caminho_tcc, "springer_segmentation_model.pkl")

# O CAMINHO PARA OS ÁUDIOS (Verifique se é este mesmo no seu computador)
pasta_audios_bmdhs = r"C:\Users\gusta\TCC\TCC\data\train" 

print(f"1/3 - A carregar o modelo Springer de: {caminho_pkl}")
modelo = joblib.load(caminho_pkl)

arquivos_wav = [f for f in os.listdir(pasta_audios_bmdhs) if f.endswith('.wav')]
print(f"2/3 - Encontrados {len(arquivos_wav)} áudios para processar.")

dados_extraidos = []

# --- 3. O LOOP DE EXTRAÇÃO ---
for arquivo in tqdm(arquivos_wav, desc="A processar pacientes"):
    caminho_audio = os.path.join(pasta_audios_bmdhs, arquivo)
    
    try:
        # Lê o áudio
        rate, audio_bruto = wav.read(caminho_audio)
        
        # SALVAGUARDA TCC: Garante que o áudio é Mono (1 canal). 
        # Se algum ficheiro do BMD-HS for estéreo, fundimos para mono para não quebrar a matemática.
        if len(audio_bruto.shape) > 1:
            audio_bruto = np.mean(audio_bruto, axis=1)

        # Normaliza o áudio
        audio_normalizado = audio_bruto.astype(np.float32)
        max_val = np.max(np.abs(audio_normalizado))
        if max_val > 0:
            audio_normalizado = audio_normalizado / max_val
            
        # A MÁGICA LIMPA: Usamos exatamente a mesma linha que funcionou no teste visual
        # O modelo processa as wavelets sozinho lá dentro!
        segmentacao = modelo.predict(audio_normalizado)
        
        # Agrupar os estados... (o código continua igual daqui para baixo)
        blocos = []
        estado_atual = segmentacao[0]
        inicio = 0
        for i in range(1, len(segmentacao)):
            if segmentacao[i] != estado_atual:
                blocos.append({'estado': estado_atual, 'inicio': inicio, 'fim': i})
                estado_atual = segmentacao[i]
                inicio = i
        blocos.append({'estado': estado_atual, 'inicio': inicio, 'fim': len(segmentacao)})
        
        # Encontrar os Ciclos Cardíacos Completos (Ordem: 1 -> 2 -> 3 -> 4)
        ciclos = []
        i = 0
        while i < len(blocos) - 3:
            if blocos[i]['estado'] == 1 and blocos[i+1]['estado'] == 2 and blocos[i+2]['estado'] == 3 and blocos[i+3]['estado'] == 4:
                ciclos.append({'S1': blocos[i], 'Sys': blocos[i+1], 'S2': blocos[i+2], 'Dia': blocos[i+3]})
                i += 4  
            else:
                i += 1
                
        TEMPO_POR_ESTADO = 0.02 # 1 / 50Hz
        Amostras_por_estado = rate / 50.0 
        
        val_RR, val_S1, val_S2, val_Sys, val_Dia = [], [], [], [], []
        val_ratio_SysRR, val_ratio_DiaRR, val_ratio_SysDia = [], [], []
        val_amp_SysS1, val_amp_DiaS2 = [], []
        
        def calc_amp_media(bloco):
            inicio_idx = int(bloco['inicio'] * Amostras_por_estado)
            fim_idx = int(bloco['fim'] * Amostras_por_estado)
            segmento = audio_normalizado[inicio_idx:fim_idx]
            if len(segmento) == 0: return 0.00001
            return np.mean(np.abs(segmento))
        
        for c in ciclos:
            # Durações
            d_S1 = (c['S1']['fim'] - c['S1']['inicio']) * TEMPO_POR_ESTADO
            d_Sys = (c['Sys']['fim'] - c['Sys']['inicio']) * TEMPO_POR_ESTADO
            d_S2 = (c['S2']['fim'] - c['S2']['inicio']) * TEMPO_POR_ESTADO
            d_Dia = (c['Dia']['fim'] - c['Dia']['inicio']) * TEMPO_POR_ESTADO
            d_RR = d_S1 + d_Sys + d_S2 + d_Dia
            
            val_S1.append(d_S1); val_Sys.append(d_Sys)
            val_S2.append(d_S2); val_Dia.append(d_Dia)
            val_RR.append(d_RR)
            
            # Razões de Tempo
            val_ratio_SysRR.append(d_Sys / d_RR if d_RR > 0 else 0)
            val_ratio_DiaRR.append(d_Dia / d_RR if d_RR > 0 else 0)
            val_ratio_SysDia.append(d_Sys / d_Dia if d_Dia > 0 else 0)
            
            # Amplitudes Absolutas
            amp_S1 = calc_amp_media(c['S1'])
            amp_Sys = calc_amp_media(c['Sys'])
            amp_S2 = calc_amp_media(c['S2'])
            amp_Dia = calc_amp_media(c['Dia'])
            
            # Razões de Amplitude
            val_amp_SysS1.append(amp_Sys / amp_S1 if amp_S1 > 0 else 0)
            val_amp_DiaS2.append(amp_Dia / amp_S2 if amp_S2 > 0 else 0)
            
        # Funções seguras
        def s_mean(lst): return np.mean(lst) if len(lst) > 0 else 0.0
        def s_std(lst): return np.std(lst, ddof=0) if len(lst) > 0 else 0.0
        
        features_paciente = {
            'patient_id': arquivo,
            'm_RR': s_mean(val_RR), 'sd_RR': s_std(val_RR),
            'm_IntS1': s_mean(val_S1), 'sd_IntS1': s_std(val_S1),
            'm_IntS2': s_mean(val_S2), 'sd_IntS2': s_std(val_S2),
            'm_IntSys': s_mean(val_Sys), 'sd_IntSys': s_std(val_Sys),
            'm_IntDia': s_mean(val_Dia), 'sd_IntDia': s_std(val_Dia),
            'm_Ratio_SysRR': s_mean(val_ratio_SysRR), 'sd_Ratio_SysRR': s_std(val_ratio_SysRR),
            'm_Ratio_DiaRR': s_mean(val_ratio_DiaRR), 'sd_Ratio_DiaRR': s_std(val_ratio_DiaRR),
            'm_Ratio_SysDia': s_mean(val_ratio_SysDia), 'sd_Ratio_SysDia': s_std(val_ratio_SysDia),
            'm_Amp_SysS1': s_mean(val_amp_SysS1), 'sd_Amp_SysS1': s_std(val_amp_SysS1),
            'm_Amp_DiaS2': s_mean(val_amp_DiaS2), 'sd_Amp_DiaS2': s_std(val_amp_DiaS2)
        }
        
        dados_extraidos.append(features_paciente)
        
    except Exception as e:
        print(f"\nErro no ficheiro {arquivo}: {e}")

# --- 4. EXPORTAÇÃO ---
print("\n3/3 - A guardar os dados...")
df_features = pd.DataFrame(dados_extraidos)
ficheiro_saida = os.path.join(caminho_tcc, "features_extraidas.csv")
df_features.to_csv(ficheiro_saida, index=False)

print(f"Sucesso absoluto! As 20 features foram guardadas no ficheiro: {ficheiro_saida}")