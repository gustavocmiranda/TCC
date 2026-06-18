# Classificação multirrótulo de doenças valvares a partir de PCG

Pipeline reprodutível para classificar quatro doenças valvares cardíacas (AS — estenose aórtica, AR — regurgitação aórtica, MR — regurgitação mitral, MS — estenose mitral) a partir de áudio de fonocardiograma (PCG) do dataset **BMD-HS**, usando *features* tabulares interpretáveis (temporais + espectrais) e classificadores clássicos, com interpretabilidade via SHAP/LIME. Trabalho de Conclusão de Curso — Ciência da Computação, UFF.

> **Foco deste README**: replicabilidade. A descrição metodológica completa está no manuscrito do TCC.

**Modelo final**: XGBoost sobre a visão combinada (áudio + demografia + contexto de captura), selecionado por uma matriz cruzada de 6 modelos × 3 visões × 20 *splits*, com validação por Wilcoxon pareado e desempate por robustez clínica por doença.

---

## 1. Pré-requisitos

| Item | Versão / requisito |
|------|--------------------|
| Python | **3.10.11** (cf. `.python-version`) |
| Sistema operacional | Linux, macOS ou Windows (caminhos via `pathlib`/`PROJECT_ROOT`) |
| RAM | ≥ 8 GB recomendado |
| Disco | ~2 GB (dataset + artefatos) |
| Tempo da modelagem (núcleo) | ~6 h em CPU (a matriz cruzada é a etapa cara) |

> **Dois caminhos de replicação** (detalhados na Seção 5):
> - **Rápido (resultados do TCC):** o `dataset_final.csv` versionado **já contém as 78 *features*** (temporais + espectrais + demografia + contexto). Basta rodar as células de modelagem (Seções 5.5–5.6) — **não é necessário baixar os áudios**.
> - **Completo (do áudio bruto):** baixar o BMD-HS e refazer segmentação, extração e fusão (Seções 5.1–5.4) antes da modelagem.

---

## 2. Instalação

### 2.1. Clonar e criar ambiente virtual

```bash
git clone https://github.com/gustavocmiranda/TCC.git
cd TCC
python3.10 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows PowerShell
pip install -r requirements.txt
```

### 2.2. Configurar o caminho do projeto via `.env`

O notebook e os módulos auxiliares leem a raiz do projeto a partir da variável de ambiente `PROJECT_ROOT`, carregada via `python-dotenv`. Copiar o template e preencher com o caminho absoluto da raiz na sua máquina:

```bash
cp .env.example .env
# editar .env e ajustar PROJECT_ROOT
```

Conteúdo esperado do `.env`:

```
PROJECT_ROOT=/caminho/absoluto/para/a/raiz/do/projeto
```

O arquivo `.env` está no `.gitignore` e não é versionado. O `.env.example` serve como modelo.

> Se `PROJECT_ROOT` não estiver definida, o notebook usa `Path.cwd()` como *fallback* — funciona desde que o Jupyter seja iniciado a partir da raiz do projeto.

---

## 3. Dados

### 3.1. Dataset BMD-HS (obrigatório apenas para o caminho completo)

Dados oficiais (download público):

**https://github.com/mHealthBuet/BMD-HS-Dataset**

Estrutura esperada após o download (criar a pasta `data/` na raiz do projeto):

```
data/
├── train.csv                       # rótulos por paciente (AS, AR, MR, MS, MD, N) + 8 recording_i
├── additional_metadata.csv         # Age, Gender, Smoker, Lives por paciente
└── train/
    ├── <patient_id>_<position>_<focus>.wav
    └── ... (872 arquivos .wav, 4 kHz, ~20 s cada)
```

Citação do dataset:

> Ali, S. N. *et al.* **BUET Multi-disease Heart Sound Dataset (BMD-HS)**. *Computer Methods and Programs in Biomedicine Update*, v. 9, p. 100237, 2026.

> Para reproduzir **apenas os resultados do TCC**, os áudios não são necessários: o `dataset_final.csv` versionado já contém todas as *features* extraídas.

### 3.2. Dados de segmentação (opcional)

A etapa de segmentação HSMM (Seção 5.1) é treinada com gravações anotadas do **PhysioNet/CinC** (https://physionet.org/). Como o modelo treinado (`springer_segmentation_model.pkl`) já está versionado, esses dados só são necessários se você quiser retreinar a segmentação do zero.

---

## 4. Estrutura do repositório

```
TCC/
├── projeto.ipynb                       # Notebook principal — pipeline ponta-a-ponta
├── source/                             # Módulos chamados pelo notebook
│   ├── extrair_features.py             # 20 features temporais (Liu et al., 2016)
│   ├── extrair_features_espectrais.py  # 49 features espectrais (MFCC + descritores + bandas clínicas)
│   ├── transformation.py               # Fusão features ↔ rótulos ↔ metadados
│   ├── feature_selection_camada1.py    # Filtro estatístico (quasi-const + Pearson + ANOVA)
│   └── training_model.py               # Baseline RF (não é o modelo final)
├── Springer_Segmentation/              # HSMM segmentation (Springer et al., 2016) — vendored
│   └── gerar_modelo.py                 # Script de treino do modelo HSMM
├── springer_segmentation_model.pkl     # Modelo HSMM pré-treinado (dispensa a Seção 5.1)
├── features_extraidas.csv              # Features temporais (dispensa a Seção 5.2)
├── dataset_final.csv                   # Dataset fundido COMPLETO: 864×84 (78 features + ids + 4 rótulos)
├── requirements.txt                    # Dependências
├── .env.example                        # Modelo de configuração (copiar para .env)
└── .python-version                     # 3.10.11
```

> O `dataset_final.csv` versionado contém **864 gravações × 84 colunas**: `patient_id`, `arquivo_wav`, os 4 rótulos (AS, AR, MR, MS) e as 78 *features* (20 temporais + 49 espectrais + 9 de demografia/contexto de captura). É o ponto de entrada do caminho rápido de replicação.

> Ao rodar o notebook, as saídas e o *checkpoint* da matriz são gravados na pasta `resultados/`, **criada automaticamente** e **não versionada** (ignorada pelo `.gitignore`). Os arquivos lá são regenerados a cada execução.

---

## 5. Pipeline de replicação

O ponto de entrada é `projeto.ipynb`. A célula de imports carrega o `.env` e define `PROJECT_ROOT`, `DATA_DIR`, `AUDIOS_DIR`, `SPRINGER_DIR` e `MODELO_PKL` — não há caminhos *hardcoded* nas células subsequentes.

As Seções 5.1–5.4 reconstroem o `dataset_final.csv` a partir do áudio bruto (**caminho completo**); todas são opcionais se você usar o `dataset_final.csv` versionado. As Seções 5.5–5.6 são o **núcleo da contribuição** e bastam para reproduzir os resultados do TCC (**caminho rápido**).

### 5.1. Segmentação HSMM (opcional — artefato já versionado)

Gera `springer_segmentation_model.pkl`. **Quando refazer**: apenas se modificar o procedimento de segmentação ou trocar o conjunto de treino (ver Seção 3.2). Tempo estimado: ~10 min.

### 5.2. Extração de *features* temporais (opcional — artefato já versionado)

Gera `features_extraidas.csv` (20 *features* de Liu et al., 2016). **Quando refazer**: se mudar o modelo HSMM ou os áudios de entrada. Tempo estimado: ~20 min para 872 áudios.

### 5.3. Extração de *features* espectrais (opcional — necessária só no caminho completo)

Gera `features_espectrais.csv` (49 *features*: MFCCs + descritores espectrais + bandas clínicas). **Não está versionado** — se for refazer a fusão do zero, rodar a célula correspondente do notebook. Tempo estimado: ~30 min para 872 áudios.

### 5.4. Fusão e construção do `dataset_final.csv` (opcional — artefato já versionado)

Une *features* temporais + espectrais + rótulos (`train.csv`) + metadados (`additional_metadata.csv`) em `dataset_final.csv`. A versão versionada **já é a versão completa (78 *features*)**, então esta etapa só é necessária no caminho completo.

### 5.5. Modelagem (obrigatória — núcleo da contribuição)

Executar as células do notebook na ordem:

| Etapa | Saída | Tempo |
|-------|-------|-------|
| Preparação dos dados (*split* paciente-level 80/20 via `iterative_train_test_split`) | divisão em memória | < 1 min |
| Feature selection Camada 1 (quasi-const + Pearson + ANOVA univariada por *label*) — aplicada em D1/D3 | *features* mantidas | < 1 min |
| **Matriz cruzada — 6 modelos × 3 visões × 20 *splits* = 360 runs** (LR, GB, RF, SVM, KNN, XGB sobre D1 áudio / D2 demografia / D3 combinado; `RandomizedSearchCV(n_iter=20)` + `GroupKFold(5)` + *threshold tuning* por *label* dentro de cada run, com *scoring* igual ao *score* clínico) | `resultados/08_matriz_runs.csv` (360 linhas) + `08_matriz_mediana_score_clinico.csv` (matriz 6×3) | ~6 h |
| **Bateria de Wilcoxon pós-matriz** — (a) pareado entre os 3 melhores em D3, (b) vencedor vs *trivial* "tudo positivo", (c.1) D3 vs D1, (c.2) D3 vs D2 | `resultados/08_wilcoxon_pos_matriz.csv` | < 1 min |
| **Diagnóstico por doença — desempate dos top-3 em D3** (max-min do pior caso por doença sob FN×5) | `resultados/08_diagnostico_top3_d3.csv` | ~5 min |

> A matriz é a etapa mais cara e pode ser **retomada a partir do *checkpoint*** `resultados/08_matriz_runs.csv` se interrompida (runs já concluídos são pulados). A pasta `resultados/` é criada automaticamente.

### 5.6. Confiabilidade e interpretabilidade (obrigatórias — análises do modelo final)

| Etapa | Saída | Tempo |
|-------|-------|-------|
| **Curvas Precision-Recall + AP + calibração (Brier, *reliability*)** — avaliação *out-of-fold* (`GroupKFold(5)`) do XGB final | `resultados/08_calibracao_brier.csv` + gráficos | ~5 min |
| **Interpretabilidade (SHAP global + local, LIME)** sobre o XGB tunado treinado em toda a base | gráficos *summary*/*waterfall* | ~10 min |
| **Importância via Regressão Logística** (*sanity check* linear cruzado contra o ranking SHAP) | `resultados/08_lr_importancia_features.csv` | ~2 min |

Os tempos são estimativas em uma CPU moderna (4 núcleos).

---

## 6. Reprodutibilidade

### 6.1. Sementes

- `random_state=42` é fixado nos **estimadores** (LR, GB, RF, SVM, XGB) e no modelo final treinado sobre toda a base.
- A matriz cruzada usa as sementes `range(100, 120)` para os **20 *splits* repetidos**. Cada semente é reaplicada igualmente aos 6 modelos × 3 visões, garantindo pareamento exato para os testes de Wilcoxon, e é usada como `random_state` do `RandomizedSearchCV` daquele run.
- `np.random.seed(seed)` é fixado imediatamente antes de `iterative_train_test_split`, garantindo o *split* multirrótulo reprodutível de cada semente.

### 6.2. Resultados esperados

Após executar a modelagem, esperar (referência: `resultados/08_matriz_mediana_score_clinico.csv` e `08_wilcoxon_pos_matriz.csv`):

- **Matriz 6×3 — mediana do *score* clínico em D3 (top-3)**: **XGB ≈ 81,50 %**, LR ≈ 81,17 %, GB ≈ 80,62 %.
- **Wilcoxon entre os 3 finalistas em D3** (empate estatístico): XGB vs LR *p* = 0,76; XGB vs GB *p* = 0,053; LR vs GB *p* = 0,90.
- **Wilcoxon vs *trivial* "tudo positivo"** (mediana 74,44 %): *p* < 0,0001, **20/20 *splits*** a favor do XGB.
- **Wilcoxon de visões**: D3 vs D1 *p* = 0,0001 (17/20 *splits*); D3 vs D2 *p* < 0,0001 (20/20).
- **Desempate por doença (max-min do pior caso)**: **XGBoost 64,81 % (em MR)** > GB 60,34 % (em AR) > LR 57,41 % (em MR) — o XGBoost não tem doença em colapso.
- **Modelo final (XGB em D3) — mediana por doença**: AS 81,48 %, AR 84,48 %, MR 77,78 %, MS 85,19 %; pior caso *macro* 74,88 %; limite inferior do IC 95 % = 66,57 %.
- **Veredito**: **XGBoost em D3** — vencedor nominal e simultaneamente o mais robusto por doença, selecionado para SHAP/LIME.

Pequenas variações (< 1 pp) podem ocorrer por diferenças de versão de `scikit-learn` ou de `xgboost`.

### 6.3. Sistema testado

- Windows 11 e Linux, Python 3.10.11. O código é independente de SO (caminhos via `pathlib` e `PROJECT_ROOT`).
- Paralelismo: `n_jobs=-1` em `RandomizedSearchCV` e nos modelos que o suportam.
