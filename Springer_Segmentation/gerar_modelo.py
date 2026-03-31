import joblib

from duration_distributions import DataDistribution
from segmentation_model import SegmentationModel
from utils import create_segmentation_array, create_train_test_split

print("--- INICIANDO TREINAMENTO DO MODELO SPRINGER ---")

# 1. COLOQUE AQUI O CAMINHO EXATO DA PASTA COM OS DADOS DO PHYSIONET (.wav e .tsv) no seu Windows
pasta_physionet = r"Springer-Segmentation\training_data" 

print("1/4 - Lendo arquivos de treinamento...")
train_recordings, train_segmentations, _, _ = create_train_test_split(
    directory=pasta_physionet,
    frac_train=1.0, 
    max_train_size=1000 
)

print(f"2/4 - Processando {len(train_recordings)} áudios e extraindo features (Isso vai demorar!)...")
clips = []
annotations = []
for i, (rec, seg) in enumerate(zip(train_recordings, train_segmentations)):
    clipped_recording, ground_truth = create_segmentation_array(
        rec, seg, recording_frequency=4000, feature_frequency=50)
    clips.extend(clipped_recording)
    annotations.extend(ground_truth)
    if (i + 1) % 10 == 0:
        print(f"      ... {i + 1} áudios processados.")

print("3/4 - Treinando o Modelo HMM/Viterbi...")
modelo_springer = SegmentationModel()
data_distribution = DataDistribution(train_segmentations)
modelo_springer.fit(clips, annotations, data_distribution=data_distribution)

print("4/4 - Salvando o modelo treinado...")
# Vai salvar o arquivo .pkl na mesma pasta onde este script está rodando
caminho_salvar = "springer_segmentation_model.pkl"
joblib.dump(modelo_springer, caminho_salvar)

print(f"--- SUCESSO! Modelo original salvo na pasta atual como: {caminho_salvar} ---")