# Classificação multirrótulo de doenças valvares a partir de PCG

Pipeline reprodutível para classificar quatro doenças valvares cardíacas (AS — estenose aórtica, AR — regurgitação aórtica, MR — regurgitação mitral, MS — estenose mitral) a partir de áudio de fonocardiograma (PCG) do dataset **BMD-HS**, usando *features* tabulares interpretáveis (temporais + espectrais) e classificadores clássicos, com interpretabilidade via SHAP/LIME. Trabalho de Conclusão de Curso — Ciência da Computação, UFF.

> **Foco deste README**: replicabilidade. A descrição metodológica completa está no manuscrito do TCC.

**Modelo final**: **Regressão Logística** sobre a visão combinada (áudio + demografia + contexto de captura), selecionada por uma matriz cruzada de 6 modelos × 3 visões × 20 *splits*, com validação por Wilcoxon pareado e desempate por robustez clínica por doença. Os três finalistas em D3 (LR, *gradient boosting* e XGBoost) **empatam estatisticamente** (Wilcoxon, *p* entre 0,41 e 0,78); a LR vence o desempate por ter o maior piso por doença sob a métrica FN×5 e oferece interpretabilidade intrínseca (coeficientes lineares).

---

## 1. Pré-requisitos

| Item | Versão / requisito |
|------|--------------------|
| Python | **3.10.11** (cf. `.python-version`) |
| Sistema operacional | **Testado apenas em Windows 11.** Espera-se portabilidade para Linux/macOS (caminhos via `pathlib`/`PROJECT_ROOT`), mas sem garantia. |
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

> **Os áudios brutos NÃO são versionados neste repositório** (a pasta `data/` está no `.gitignore`, por se tratar de dados de terceiros sob licença própria). Para o **caminho rápido** eles são dispensáveis — veja a nota no fim desta seção. Só baixe o BMD-HS se for refazer a extração de *features* a partir do áudio.

Download público (espelho atual, com a estrutura já organizada):

**https://github.com/sani002/BMD-HS-Dataset**

> O repositório original dos autores (`github.com/mHealthBuet/BMD-HS-Dataset`), citado no artigo, está **fora do ar (HTTP 404)** em junho/2026; o espelho acima contém os mesmos 872 áudios, `train.csv` e `additional_metadata.csv`. A referência acadêmica canônica continua sendo o artigo do dataset (citação abaixo).

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

A etapa de segmentação HSMM (Seção 5.1) é treinada com gravações anotadas do **CirCor DigiScope Phonocardiogram Dataset** (dados de treino do PhysioNet/CinC Challenge 2022), download público:

**https://physionet.org/content/circor-heart-sound/1.0.3/** (DOI: [10.13026/tshs-mw03](https://doi.org/10.13026/tshs-mw03))

```bash
# baixar via wget (ou usar o ZIP de ~450 MB disponível na página)
wget -r -N -c -np https://physionet.org/files/circor-heart-sound/1.0.3/
```

Como o modelo treinado (`springer_segmentation_model.pkl`) já está versionado, esses dados só são necessários se você quiser **retreinar a segmentação do zero** (Seção 5.1). Os áudios devem ser apontados em `Springer_Segmentation/training_data/` (também ignorado pelo `.gitignore`).

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
│   └── training_model.py               # Baseline RF (auxiliar — não é o modelo final)
├── Springer_Segmentation/              # HSMM segmentation (Springer et al., 2016) — vendored
│   └── gerar_modelo.py                 # Script de treino do modelo HSMM
├── springer_segmentation_model.pkl     # Modelo HSMM pré-treinado (dispensa a Seção 5.1)
├── features_extraidas.csv              # Features temporais (dispensa a Seção 5.2)
├── features_espectrais.csv             # Features espectrais (dispensa a Seção 5.3)
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

### 5.3. Extração de *features* espectrais (opcional — artefato já versionado)

Gera `features_espectrais.csv` (49 *features*: MFCCs + descritores espectrais + bandas clínicas). **Quando refazer**: se mudar o modelo HSMM ou os áudios de entrada. Tempo estimado: ~30 min para 872 áudios.

### 5.4. Fusão e construção do `dataset_final.csv` (opcional — artefato já versionado)

Une *features* temporais + espectrais + rótulos (`train.csv`) + metadados (`additional_metadata.csv`) em `dataset_final.csv`. A versão versionada **já é a versão completa (78 *features*)**, então esta etapa só é necessária no caminho completo.

### 5.5. Modelagem (obrigatória — núcleo da contribuição)

Executar as células do notebook na ordem:

| Etapa | Saída | Tempo |
|-------|-------|-------|
| Preparação dos dados (*split* paciente-level 80/20 via `iterative_train_test_split`) | divisão em memória | < 1 min |
| Feature selection Camada 1 (quasi-const + Pearson + ANOVA univariada por *label*) — aplicada em D1/D3 | *features* mantidas | < 1 min |
| **Matriz cruzada — 6 modelos × 3 visões × 20 *splits* = 360 runs** (LR, GB, RF, SVM, KNN, XGB sobre D1 áudio / D2 demografia / D3 combinado; `RandomizedSearchCV(n_iter=20)` + `GroupKFold(5)` + *threshold tuning* por *label* dentro de cada run, com *scoring* igual ao *score* clínico) | `resultados/matriz_runs.csv` (360 linhas) + `matriz_mediana_score_clinico.csv` (matriz 6×3) | ~6 h |
| **Bateria de Wilcoxon pós-matriz** — (a) pareado entre os 3 melhores em D3, (b) vencedor vs *trivial* "tudo positivo", (c.1) D3 vs D1, (c.2) D3 vs D2 | `resultados/wilcoxon_pos_matriz.csv` | < 1 min |
| **Diagnóstico por doença — desempate dos top-3 em D3** (max-min do pior caso por doença sob FN×5) | `resultados/diagnostico_top3_d3.csv` | ~5 min |

> A matriz é a etapa mais cara e pode ser **retomada a partir do *checkpoint*** `resultados/matriz_runs.csv` se interrompida (runs já concluídos são pulados). A pasta `resultados/` é criada automaticamente.

### 5.6. Confiabilidade e interpretabilidade (obrigatórias — análises do modelo final)

| Etapa | Saída | Tempo |
|-------|-------|-------|
| **Importância global pelos coeficientes da LR** — como o modelo final é linear, sua regra de decisão *são* os coeficientes (sinal + magnitude por *feature* e por doença, sobre *features* padronizadas); interpretação global primária, sem explicador *post-hoc* | `resultados/lr_importancia_features.csv` + figura de barras | ~2 min |
| **Verificação cruzada de família — SHAP no XGBoost companheiro** (confirma as *features* dominantes em um modelo de árvore) | gráficos *summary* | ~10 min |
| **Explicação local (SHAP `LinearExplainer` + LIME) sobre a própria LR** — TP de alta confiança e FN *borderline* por doença | gráficos *waterfall* | ~5 min |
| **Curvas Precision-Recall + AP + calibração (Brier, *reliability*)** — avaliação *out-of-fold* (`GroupKFold(5)`) da LR final | `resultados/calibracao_brier.csv` + gráficos | ~5 min |

Os tempos são estimativas medidas no sistema descrito na Seção 6.3.

---

## 6. Reprodutibilidade

### 6.1. Sementes

- `random_state=42` é fixado nos **estimadores** (LR, GB, RF, SVM, XGB) e no modelo final treinado sobre toda a base.
- A matriz cruzada usa as sementes `range(100, 120)` para os **20 *splits* repetidos**. Cada semente é reaplicada igualmente aos 6 modelos × 3 visões, garantindo pareamento exato para os testes de Wilcoxon, e é usada como `random_state` do `RandomizedSearchCV` daquele run.
- `np.random.seed(seed)` é fixado imediatamente antes de `iterative_train_test_split`, garantindo o *split* multirrótulo reprodutível de cada semente.
- **Treino da segmentação HSMM** (`Springer_Segmentation/gerar_modelo.py`): `seed=42` fixa `random.seed` **e** `np.random.seed` antes do treino, cobrindo os dois pontos estocásticos do ajuste (`random.shuffle` em `create_train_test_split` e `np.random.permutation` na subamostragem de estados em `segmentation_model.fit`). Em `utils.py`, a lista de pacientes é ordenada (`sorted`) antes do *shuffle*, eliminando a dependência do `PYTHONHASHSEED`. Com isso, dois treinos do modelo de segmentação geram um `.pkl` **byte-idêntico** (verificado).

### 6.2. Resultados esperados

Após executar a modelagem, esperar (referência: o manuscrito do TCC; localmente, `resultados/matriz_mediana_score_clinico.csv` e `wilcoxon_pos_matriz.csv`):

- **Matriz 6×3 — mediana do *score* clínico em D3 (top-3)**: **LR ≈ 81,03 %**, GB ≈ 80,14 %, XGB ≈ 79,81 %.
- **Wilcoxon entre os 3 finalistas em D3** (empate estatístico): LR vs GB *p* = 0,78; LR vs XGB *p* = 0,41; GB vs XGB *p* = 0,50.
- **Wilcoxon vs *trivial* "tudo positivo"** (mediana 74,44 %): *p* < 0,0001, **18/20 *splits*** a favor da LR.
- **Wilcoxon de visões**: D3 vs D1 (ausculta, 77,24 %) *p* = 0,0083 (15/20 *splits*); D3 vs D2 (demografia, 75,37 %) *p* = 0,0009 (17/20).
- **Desempate por doença (max-min do pior caso)**: **LR 57,41 % (em MR)** > XGB 53,70 % (em MR) > GB 51,85 % (em MR); a LR também tem o maior IC 95 % inferior na pior doença (60,93 %) — é a que melhor protege o piso por doença.
- **Modelo final (LR em D3) — mediana por doença**: AS 86,00 %, AR 82,76 %, MS 75,93 %, MR 79,63 %; pior caso *macro* 71,92 %; limite inferior do IC 95 % do *macro* = 72,40 %.
- **Confiabilidade (*out-of-fold*)**: AP por doença de 0,533 (MR) a 0,701 (AS) — AR 0,574, MS 0,578; *Brier score* de 0,166 (AS) a 0,229 (AR), com diagramas de confiabilidade próximos da diagonal (probabilidades bem calibradas, sem recalibração).
- **Veredito**: **Regressão Logística em D3** — vencedora nominal e simultaneamente a mais robusta por doença, escolhida como modelo final pela interpretabilidade intrínseca.

> **Sobre a variação entre execuções.** Os três finalistas (LR, GB, XGB) estão em **empate estatístico** (Wilcoxon, *p* = 0,41–0,78). Como a diferença de mediana entre eles é da ordem de ~1 pp, o **ranking nominal e os dígitos exatos podem oscilar** entre versões de `scikit-learn`/`xgboost` ou hardware — em alguns ambientes o XGBoost chega a liderar nominalmente. O que se reproduz de forma robusta é a **metodologia** (matriz → Wilcoxon → desempate por pior caso por doença sob FN×5) e o **empate** em si; os números acima são os reportados no TCC. A escolha da LR como modelo final não depende de uma diferença significativa de desempenho, e sim do critério de robustez por doença somado à interpretabilidade.

### 6.3. Sistema testado

Pipeline executado e validado em uma única configuração:

- **Sistema operacional**: Windows 11. O código não foi testado em Linux ou macOS; espera-se portabilidade (caminhos via `pathlib`/`PROJECT_ROOT`), mas sem garantia.
- **Python**: 3.10.11.
- **CPU**: Intel Core i5-10400F (6 núcleos / 12 *threads*).
- **Memória**: 16 GB de RAM.

### 6.4. Regeneração completa do zero

Os artefatos versionados (`*.pkl`, `*.csv`) permitem reproduzir os resultados sem
recomputar. Para **regenerar tudo a partir do áudio bruto** — gerando novo modelo de
segmentação, novas *features* e nova matriz — é preciso **apagar os artefatos
regeneráveis antes de rodar o notebook**, pois há *guards* que pulam etapas cujo
arquivo de saída já existe (inclusive o *checkpoint* da matriz, que retoma runs já
concluídos). Com as sementes fixas (Seção 6.1), o resultado é determinístico.

Apagar **apenas** os arquivos abaixo (os dados brutos em `data/` e
`Springer_Segmentation/training_data/` **nunca** são apagados):

```bash
rm -f springer_segmentation_model.pkl \
      features_extraidas.csv \
      features_espectrais.csv \
      dataset_final.csv
rm -rf resultados/          # inclui o checkpoint matriz_runs.csv
```

Em seguida, executar o `projeto.ipynb` do início ao fim (*Run All*). A ordem de
regeneração é: segmentação HSMM → *features* temporais/espectrais → `dataset_final.csv`
→ matriz cruzada → análises. Tempo total dominado pela matriz (~6 h em CPU).
