import os
import sys
import joblib
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from scipy.signal import welch
import librosa
from tqdm.notebook import tqdm


BANDAS_CLINICAS = {
    'band_25_45':   (25, 45),
    'band_45_80':   (45, 80),
    'band_80_150':  (80, 150),
    'band_150_300': (150, 300),
    'band_300_600': (300, 600),
}


def _energias_bandas(audio, sr):
    if len(audio) < 64:
        return {k: 0.0 for k in BANDAS_CLINICAS}
    nperseg = min(256, len(audio))
    freqs, psd = welch(audio, fs=sr, nperseg=nperseg)
    total = psd.sum() + 1e-12
    return {
        nome: float(psd[(freqs >= lo) & (freqs < hi)].sum() / total)
        for nome, (lo, hi) in BANDAS_CLINICAS.items()
    }


def _concat_fases(audio, segmentacao, rate):
    amostras_por_estado = rate / 50.0
    blocos = []
    estado, ini = segmentacao[0], 0
    for i in range(1, len(segmentacao)):
        if segmentacao[i] != estado:
            blocos.append((estado, ini, i))
            estado, ini = segmentacao[i], i
    blocos.append((estado, ini, len(segmentacao)))

    sis, dia = [], []
    for est, i0, i1 in blocos:
        s = int(i0 * amostras_por_estado)
        e = int(i1 * amostras_por_estado)
        if est == 2:
            sis.append(audio[s:e])
        elif est == 4:
            dia.append(audio[s:e])
    return (
        np.concatenate(sis) if sis else np.array([], dtype=np.float32),
        np.concatenate(dia) if dia else np.array([], dtype=np.float32),
    )


def extrair_features_espectrais(caminho_repo, caminho_modelo, pasta_audios, arquivo_saida):
    """
    Extrai features espectrais por gravacao, ancoradas na segmentacao Springer.

    Features geradas (~49):
      - MFCCs (13 coef.) media + desvio                               -> 26
      - Centroide, bandwidth, rolloff, flatness, ZCR (media + desvio) -> 10
      - Energia relativa em 5 bandas clinicas (sinal inteiro)          ->  5
      - Razao Sistole/Diastole em cada banda clinica                   ->  5
      - Centroide espectral medio em Sistole e em Diastole             ->  2
      - Razao de energia total Sistole/Diastole                        ->  1
    """
    print("--- INICIANDO EXTRACAO DE FEATURES ESPECTRAIS ---")

    if caminho_repo not in sys.path:
        sys.path.insert(0, caminho_repo)

    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Modelo nao encontrado: {caminho_modelo}")

    print("1/3 - Carregando modelo Springer...")
    modelo = joblib.load(caminho_modelo)

    arquivos = sorted([f for f in os.listdir(pasta_audios) if f.endswith('.wav')])
    print(f"2/3 - Processando {len(arquivos)} audios em {os.path.basename(pasta_audios)}")

    n_fft = 512
    hop = 128
    dados = []

    for arquivo in tqdm(arquivos, desc="Features espectrais"):
        caminho = os.path.join(pasta_audios, arquivo)
        try:
            rate, audio = wav.read(caminho)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            segmentacao = modelo.predict(audio)
            sis_audio, dia_audio = _concat_fases(audio, segmentacao, rate)

            feats = {'patient_id': arquivo}

            mfcc = librosa.feature.mfcc(
                y=audio, sr=rate, n_mfcc=13,
                n_fft=n_fft, hop_length=hop, n_mels=40, fmax=rate / 2,
            )
            for i in range(13):
                feats[f'mfcc_m_{i+1}'] = float(np.mean(mfcc[i]))
                feats[f'mfcc_sd_{i+1}'] = float(np.std(mfcc[i]))

            cent = librosa.feature.spectral_centroid(y=audio, sr=rate, n_fft=n_fft, hop_length=hop)[0]
            bw = librosa.feature.spectral_bandwidth(y=audio, sr=rate, n_fft=n_fft, hop_length=hop)[0]
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=rate, n_fft=n_fft, hop_length=hop)[0]
            flat = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop)[0]
            zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop)[0]

            for arr, nome in [(cent, 'cent'), (bw, 'bw'), (rolloff, 'rolloff'),
                              (flat, 'flat'), (zcr, 'zcr')]:
                feats[f'm_{nome}'] = float(np.mean(arr))
                feats[f'sd_{nome}'] = float(np.std(arr))

            for banda, valor in _energias_bandas(audio, rate).items():
                feats[banda] = valor

            ener_sis = _energias_bandas(sis_audio, rate)
            ener_dia = _energias_bandas(dia_audio, rate)
            for banda in BANDAS_CLINICAS:
                feats[f'ratio_sd_{banda}'] = ener_sis[banda] / (ener_dia[banda] + 1e-6)

            if len(sis_audio) >= 256:
                cs = librosa.feature.spectral_centroid(
                    y=sis_audio, sr=rate, n_fft=256, hop_length=64,
                )[0]
                feats['cent_sis'] = float(np.mean(cs))
            else:
                feats['cent_sis'] = 0.0

            if len(dia_audio) >= 256:
                cd = librosa.feature.spectral_centroid(
                    y=dia_audio, sr=rate, n_fft=256, hop_length=64,
                )[0]
                feats['cent_dia'] = float(np.mean(cd))
            else:
                feats['cent_dia'] = 0.0

            e_sis = float(np.sum(sis_audio ** 2)) if len(sis_audio) else 0.0
            e_dia = float(np.sum(dia_audio ** 2)) if len(dia_audio) else 0.0
            feats['energia_ratio_sd'] = e_sis / (e_dia + 1e-6)

            dados.append(feats)

        except Exception as e:
            print(f"Erro em {arquivo}: {e}")

    df = pd.DataFrame(dados)
    df.to_csv(arquivo_saida, index=False)
    print(f"3/3 - Salvo em: {os.path.basename(arquivo_saida)}  ({df.shape[0]} gravacoes x {df.shape[1]-1} features)")
    return df
