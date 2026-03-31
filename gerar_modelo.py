import joblib
from Springer_Segmentation.duration_distributions import DataDistribution
from Springer_Segmentation.segmentation_model import SegmentationModel
from Springer_Segmentation.utils import create_segmentation_array, create_train_test_split

# 1. Puxa a amostra do PhysioNet (com os arquivos .tsv)
train_recordings, train_segmentations, _, _ = create_train_test_split(
    directory="data/generate_model_data", 
    frac_train=1.0, # Usa 100% da amostra para treinar
    max_train_size=1000
)

# 2. Prepara os áudios (exatamente como no full_training_script.py deles)
clips = []
annotations = []
for rec, seg in zip(train_recordings, train_segmentations):
    clipped_recording, ground_truth = create_segmentation_array(
        rec, seg, recording_frequency=4000, feature_frequency=50)
    clips.extend(clipped_recording)
    annotations.extend(ground_truth)

# 3. Treina o Modelo (O "Fit" acontece aqui, só desta vez)
modelo_springer = SegmentationModel()
data_distribution = DataDistribution(train_segmentations)
modelo_springer.fit(clips, annotations, data_distribution=data_distribution)

# 4. A MÁGICA: Salva o modelo treinado no seu computador!
joblib.dump(modelo_springer, 'model/meu_springer_treinado.pkl')
print("Modelo treinado e salvo com sucesso!")