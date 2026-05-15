# Classificação multilabel de doenças valvares a partir de PCG

Pipeline reprodutível para classificar quatro doenças valvares cardíacas (AS, AR, MR, MS) a partir de áudio de fonocardiograma do dataset **BMD-HS**, usando *features* tabulares (temporais + espectrais) e XGBoost com interpretabilidade via SHAP/LIME. Trabalho de Conclusão de Curso — Ciência da Computação, UFF.

> **Foco deste README**: replicabilidade. A descrição metodológica completa está no manuscrito do TCC.

---

## 1. Pré-requisitos

| Item | Versão / requisito |
|------|--------------------|
| Python | **3.10.11** (cf. `.python-version`) |
| Sistema operacional | Linux, macOS ou Windows |
| RAM | ≥ 8 GB recomendado |
| Disco | ~2 GB (dataset + artefatos) |
| Tempo de execução end-to-end | ~2–4 h (CPU) |

> Os artefatos mais custosos (`springer_segmentation_model.pkl`, `features_extraidas.csv`, `dataset_final.csv`) já estão versionados no repositório. Quem só quiser reproduzir a parte de modelagem pode pular direto para a Seção 5.3.

---

## 2. Instalação

### 2.1. Clonar e criar ambiente virtual

```bash
git clone <URL_DO_REPO> TCC
cd TCC
python3.10 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows PowerShell
```

### 2.2. Converter `requirements.txt` (necessário em Linux/macOS)

O arquivo está em **UTF-16 LE com CRLF** (gerado por `pip freeze` no PowerShell). `pip install -r` em Linux/macOS não interpreta esse encoding. Converter para UTF-8 antes de instalar:

```bash
iconv -f UTF-16LE -t UTF-8 requirements.txt | tr -d '\r' > requirements_utf8.txt
pip install -r requirements_utf8.txt
```

No Windows o arquivo original funciona sem conversão.

### 2.3. Dependências-chave

`scikit-learn==1.7.2`, `xgboost==3.2.0`, `scikit-multilearn==0.2.0`, `librosa==0.11.0`, `shap==0.49.1`, `lime==0.2.0.1`, `numpy==2.2.6`, `pandas==2.3.3`, `scipy==1.15.3`. Versões diferentes de `scikit-learn` e `xgboost` podem alterar resultados numéricos por mudanças internas de algoritmo.

---

## 3. Dataset BMD-HS

O dataset não está versionado (excluído via `.gitignore`). Baixar de:

**https://github.com/mHealthBuet/BMD-HS-Dataset**

Estrutura esperada após download (criar a pasta `data/` na raiz do projeto):

```
data/
├── train.csv                       # rótulos por paciente (AS, AR, MR, MS, MD, N) + 8 recording_i
├── additional_metadata.csv         # Age, Gender, Smoker, Lives por paciente
└── train/
    ├── <patient_id>_<position>_<focus>.wav
    └── ... (872 arquivos .wav, 4 kHz, ~20 s cada)
```

Citação do dataset: Ali, S. N. *et al.* (2024). *BUET Multi-disease Heart Sound Dataset*. Computer Methods and Programs in Biomedicine Update, v. 9, 100237. doi:10.1016/j.cmpbup.2026.100237.

---

## 4. Estrutura do repositório

```
TCC/
├── projeto.ipynb                       # Notebook principal — pipeline ponta-a-ponta
├── source/                             # Módulos chamados pelo notebook
│   ├── extrair_features.py             # 20 features temporais (Liu et al., 2016)
│   ├── extrair_features_espectrais.py  # 49 features espectrais (MFCC + MPEG-7 + bandas clínicas)
│   ├── transformation.py               # Fusão features ↔ rótulos ↔ metadados
│   ├── feature_selection_camada1.py    # Filtro estatístico (quasi-const + Pearson + ANOVA)
│   └── training_model.py               # Baseline RF (não é o modelo final)
├── Springer_Segmentation/              # HSMM segmentation (Springer et al., 2016) — vendored
│   └── gerar_modelo.py                 # Script de treino do modelo HSMM
├── springer_segmentation_model.pkl     # Modelo HSMM pré-treinado (pular Seção 5.1)
├── features_extraidas.csv              # Features temporais (pular Seção 5.2)
├── dataset_final.csv                   # Dataset fundido (pular Seção 5.4)
├── requirements.txt                    # Dependências (UTF-16 LE — ver Seção 2.2)
└── .python-version                     # 3.10.11
```

---

## 5. Pipeline de replicação

O ponto de entrada é `projeto.ipynb`. Antes de executar, **ajustar a constante `CAMINHO_TCC`** em todas as células que a definem (atualmente codificada como `r"C:\Users\gusta\TCC\TCC"`). Substituir pelo caminho absoluto da raiz do projeto na sua máquina. Em Linux/macOS, trocar também `r"...\data\train"` por `os.path.join(CAMINHO_TCC, "data", "train")`.

A ordem das etapas abaixo segue a ordem das células do notebook. Cada etapa é independente — pode-se pular as primeiras três se os artefatos versionados forem usados como ponto de partida.

### 5.1. Segmentação HSMM (opcional — já realizada)

Gera `springer_segmentation_model.pkl`. Já incluído no repositório.

**Quando refazer**: apenas se modificar o procedimento de segmentação ou trocar o conjunto de treino. Requer dados do PhysioNet/CinC Challenge 2022 colocados em `Springer_Segmentation/training_data/`.

Tempo estimado: ~10 min.

### 5.2. Extração de *features* temporais (opcional — já realizada)

Gera `features_extraidas.csv` (20 *features* de Liu et al., 2016). Já incluído.

**Quando refazer**: se mudar o modelo HSMM ou os áudios de entrada.

Tempo estimado: ~20 min para 872 áudios.

### 5.3. Extração de *features* espectrais (obrigatória se for refazer a fusão)

Gera `features_espectrais.csv` (49 *features*: 26 MFCCs + 10 MPEG-7 + 13 de bandas clínicas). **Não está versionado** — precisa ser gerado localmente:

```python
# Célula correspondente do notebook (após ajuste de CAMINHO_TCC)
features_espectrais = extrair_features_espectrais(
    caminho_repo=PASTA_REPO,
    caminho_modelo=MODELO_PKL,
    pasta_audios=PASTA_AUDIOS,
    arquivo_saida="features_espectrais.csv",
)
```

Tempo estimado: ~30 min para 872 áudios.

### 5.4. Fusão e construção do `dataset_final.csv` (opcional — já realizada)

Une *features* temporais + espectrais + rótulos (`train.csv`) + metadados (`additional_metadata.csv`). A versão versionada no repositório **inclui apenas as features temporais**. Para incluir as espectrais, rodar a célula de merge das features espectrais no notebook.

### 5.5. Modelagem (obrigatória — núcleo da contribuição)

Executar as células do notebook na ordem:

| Etapa | Saída | Tempo |
|-------|-------|-------|
| Preparação dos dados (split paciente-level 80/20 via `iterative_train_test_split`) | divisão em memória | < 1 min |
| Feature selection Camada 1 (quasi-const + Pearson + ANOVA univariada por *label*) | ~64 *features* mantidas | < 1 min |
| *Threshold tuning* por *label* (GroupKFold 5) | *thresholds* otimizados por doença | ~5 min |
| Benchmark de 6 modelos (LR, GB, RF, SVM, KNN, XGB) com CV aninhada | tabela de *scores* | ~20 min |
| *Tuning* de hiperparâmetros (GridSearchCV XGB + RF) | melhores hiperparâmetros | ~30 min |
| Diagnóstico por *label* (XGBoost tunado) | tabela per-doença | ~3 min |
| Avaliação no *held-out* (XGB vs RF vs *trivial*) | tabela de *scores* | ~5 min |
| **Validação por 20 *splits* (Wilcoxon pareado)** | decisão estatística | ~60 min |
| **Interpretabilidade (SHAP global + local, LIME)** | gráficos *waterfall*, *summary plot* | ~10 min |

Os tempos são estimativas em uma CPU moderna (4 núcleos). Resultados intermediários são salvos em `resultados/` (criada automaticamente).

---

## 6. Reprodutibilidade

### 6.1. Sementes

- `random_state=42` é o padrão em todo o *pipeline* (`train_test_split`, modelos, `GridSearchCV`).
- Para a validação por *splits* repetidos (Etapa 12), as sementes são `range(100, 120)` para a rodada principal e `range(0, 20)` para a rodada de robustez. Ambas chegam ao mesmo veredito (XGB > RF).
- `np.random.seed(42)` é fixado imediatamente antes de `iterative_train_test_split` para reprodutibilidade do *split* multilabel.

### 6.2. Resultados esperados

Após executar todo o *pipeline*, esperar:

- **Mediana do *score* clínico macro nos 20 *splits*** (XGBoost): ~81 %
- **Wilcoxon pareado XGB vs RF**: p ≈ 0,003
- ***Held-out* macro (XGBoost tunado)**: ~83 %

Pequenas variações (< 1 pp) podem ocorrer por diferenças de versão de `scikit-learn` ou de `xgboost`.

### 6.3. Sistema testado

- Windows 11, Python 3.10.11
- Núcleos da CPU usados para paralelismo: `n_jobs=-1` em `GridSearchCV` e modelos que suportam.

---

## 7. Solução de problemas comuns

| Sintoma | Causa provável | Solução |
|---------|----------------|---------|
| `pip install -r requirements.txt` lança `UnicodeDecodeError` | Arquivo em UTF-16 LE | Ver Seção 2.2 (`iconv`) |
| Notebook quebra em `FileNotFoundError: data/train.csv` | `data/` não baixado | Ver Seção 3 |
| Notebook quebra em `pasta_audios não existe` | `CAMINHO_TCC` ainda aponta para o caminho original do autor | Ajustar a constante em **todas** as células que a definem |
| Em Linux/macOS, `r'data\train.csv'` falha | Separador de path Windows | Trocar para `os.path.join('data', 'train.csv')` em `source/transformation.py` linhas 11–12 |
| `ImportError: No module named 'Springer_Segmentation'` | `sys.path` não inclui `CAMINHO_TCC` | Os scripts já fazem `sys.path.insert(0, caminho_repo)`; se ainda falhar, rodar o notebook a partir da raiz do projeto |
| Resultados do XGBoost levemente diferentes | Versão de `xgboost` diferente de 3.2.0 | Fixar a versão (`pip install xgboost==3.2.0`) |
| `shap.TreeExplainer` lança `Additivity check failed` | Imprecisão numérica em estimadores específicos | O notebook já trata com `try/except` e `check_additivity=False` como *fallback* |

---

## 8. Licença

CC BY 4.0 (segue a licença do dataset BMD-HS). Ver `LICENSE`.

---

## 9. Citação

Caso utilize este pipeline, citar o trabalho original do TCC (em preparação) e o dataset:

```bibtex
@article{ALI2026100237,
  title  = {BUET multi-disease heart sound dataset: A comprehensive auscultation dataset for developing computer-aided diagnostic systems},
  author = {Ali, Shams Nafisa and Zahin, Afia and Shuvo, Samiul Based and others},
  journal = {Computer Methods and Programs in Biomedicine Update},
  volume = {9},
  pages  = {100237},
  year   = {2024},
  doi    = {10.1016/j.cmpbup.2026.100237}
}
```
