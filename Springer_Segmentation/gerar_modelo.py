import sys
import random
import joblib
import numpy as np
from tqdm.notebook import tqdm

def treinar_modelo_springer(caminho_repo, pasta_dados, arquivo_saida, max_audios=None, seed=42):
    """
    Treina o modelo de segmentação de Springer (HMM/Viterbi) a partir dos dados do PhysioNet.
    Retorna o modelo treinado e o salva em um arquivo .pkl.

    Reprodutibilidade: o treino é estocástico em dois pontos que usam o RNG global
    (random.shuffle em utils.create_train_test_split e np.random.permutation na
    subamostragem de estados em segmentation_model.fit). Fixamos `seed` em ambos os
    RNGs para garantir que retreinos futuros sejam determinísticos.
    NOTA: o .pkl versionado no repositório (usado no paper) foi gerado ANTES desta
    correção, sem semente; portanto um retreino seeded NÃO reproduz aquele artefato
    byte a byte. O .pkl canônico é a fonte da verdade — o guard `if not MODELO_PKL.exists()`
    no notebook impede sobrescrevê-lo por acidente.
    """
    print("--- INICIANDO TREINAMENTO DO MODELO SPRINGER ---")

    # Fixa os dois RNGs globais ANTES de qualquer chamada que os consome.
    random.seed(seed)
    np.random.seed(seed)

    # 1. Injeção do caminho (Essencial no Jupyter para não dar erro de ModuleNotFound)
    if caminho_repo not in sys.path:
        sys.path.insert(0, caminho_repo)

    # Importações específicas do Springer (agora seguras)
    from duration_distributions import DataDistribution
    from segmentation_model import SegmentationModel
    from utils import create_segmentation_array, create_train_test_split

    # 2. Carregar os gabaritos e áudios
    print("1/4 - Lendo arquivos de treinamento (.wav e .tsv)...")
    train_recordings, train_segmentations, _, _ = create_train_test_split(
        directory=pasta_dados,
        frac_train=1.0, 
        max_train_size=max_audios 
    )

    # 3. Processamento e Extração Interna
    print(f"2/4 - Processando {len(train_recordings)} áudios e extraindo features (Isso vai demorar!)...")
    clips = []
    annotations = []
    
    # Loop com a barra de progresso interativa
    for rec, seg in tqdm(zip(train_recordings, train_segmentations), total=len(train_recordings), desc="Processando Áudios"):
        clipped_recording, ground_truth = create_segmentation_array(
            rec, seg, recording_frequency=4000, feature_frequency=50)
        clips.extend(clipped_recording)
        annotations.extend(ground_truth)

    # 4. Treinamento pesado
    print("\n3/4 - Treinando o Modelo HMM/Viterbi...")
    modelo_springer = SegmentationModel()
    data_distribution = DataDistribution(train_segmentations)
    modelo_springer.fit(clips, annotations, data_distribution=data_distribution)

    # 5. Exportação
    print("4/4 - Salvando o modelo treinado...")
    joblib.dump(modelo_springer, arquivo_saida)

    print("--- SUCESSO! Modelo Springer treinado e salvo. ---")
    
    # Retorna o modelo em memória caso queira usar na mesma hora no Notebook
    return modelo_springer