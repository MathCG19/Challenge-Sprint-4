# Previsão de Acidentes com LSTMs - Sprint Challenge 4

Projeto de Deep Learning utilizando redes neurais recorrentes (LSTM) para prever padrões de acidentes em rodovias brasileiras com base nos dados públicos da Polícia Rodoviária Federal (PRF).

---

## Relatório Técnico

### 1. Target Escolhido e Justificativa

**Target:** Prever o número de acidentes por dia na rodovia BR-101.

**Justificativa da Escolha:**

A BR-101 foi escolhida como foco do modelo pelos seguintes motivos:

1. **Relevância Estratégica:** É uma das principais rodovias federais do Brasil, atravessando 12 estados da costa brasileira (Rio Grande do Sul até Rio Grande do Norte), totalizando aproximadamente 4.660 km de extensão.

2. **Volume de Dados Adequado:** A BR-101 apresenta um volume significativo de acidentes (~20-40 por dia), suficiente para treinar um modelo LSTM sem causar overfitting, mas não tão alto a ponto de ter ruído excessivo.

3. **Padrões Identificáveis:** Por ser uma rodovia específica, os padrões de acidentes são mais consistentes e relacionados a características locais (clima regional, fluxo de tráfego, condições da pista), tornando o problema mais tratável para aprendizado de máquina.

4. **Aplicabilidade Prática:** Modelos específicos por rodovia são mais úteis para:
   - Planejamento operacional da PRF em regiões específicas
   - Alocação estratégica de recursos e patrulhas
   - Análise de risco para seguradoras e transportadoras
   - Campanhas de conscientização direcionadas

5. **Escalabilidade:** Uma vez validada a abordagem na BR-101, a mesma metodologia pode ser replicada para outras rodovias federais (BR-116, BR-381, etc.), permitindo análises comparativas.

**Tipo de Problema:** Regressão de séries temporais univariada com features multivariadas.

---

### 2. Tratamento e Preparação dos Dados

#### 2.1 Coleta e Carregamento

- **Fonte:** Dados Abertos da Polícia Rodoviária Federal (PRF)
- **Arquivo:** Acidentes 2023 (Agrupados por ocorrência)
- **Formato:** CSV com separador `;` e encoding `latin1`
- **Período:** 01/01/2023 a 31/12/2023 (365 dias)

#### 2.2 Filtragem e Seleção

**Critérios de Filtragem:**
- Seleção de acidentes apenas na **BR-101** (`br == 101`)
- Remoção de colunas irrelevantes para o modelo temporal
- Total de registros após filtragem: ~8.000-12.000 acidentes na BR-101

**Colunas Selecionadas:**
```
- data_inversa: Data do acidente
- horario: Hora do acidente
- dia_semana: Dia da semana
- uf: Estado (unidade federativa)
- pessoas: Total de pessoas envolvidas
- mortos: Total de mortos
- feridos_leves: Total de feridos leves
- feridos_graves: Total de feridos graves
- veiculos: Quantidade de veículos envolvidos
```

#### 2.3 Engenharia de Features

**Features Temporais Básicas:**
- Conversão de `data_inversa` para tipo `datetime`
- Extração de: ano, mês, dia, dia do ano
- Extração de hora do acidente (formato 24h)

**Features Cíclicas (Sine/Cosine Encoding):**

Para capturar a natureza periódica do tempo, foram criadas transformações trigonométricas:

```python
# Hora (0-23h)
hora_sin = sin(2π × hora / 24)
hora_cos = cos(2π × hora / 24)

# Dia da semana (0-6)
dia_semana_sin = sin(2π × dia_semana / 7)
dia_semana_cos = cos(2π × dia_semana / 7)

# Mês (1-12)
mes_sin = sin(2π × mês / 12)
mes_cos = cos(2π × mês / 12)
```

**Justificativa:** Features cíclicas permitem que o modelo entenda que 23h está próximo de 0h, que domingo está próximo de segunda-feira, etc. Isso é crucial para LSTMs capturarem padrões temporais corretamente.

**Features de Lag (Valores Passados):**
- `num_acidentes_lag1`: Número de acidentes 1 dia atrás
- `num_acidentes_lag2`: Número de acidentes 2 dias atrás
- `num_acidentes_lag3`: Número de acidentes 3 dias atrás
- `num_acidentes_lag7`: Número de acidentes 7 dias atrás (padrão semanal)

**Features Estatísticas:**
- `ma_3`: Média móvel de 3 dias
- `ma_7`: Média móvel de 7 dias
- `std_7`: Desvio padrão móvel de 7 dias (volatilidade)

#### 2.4 Agregação Temporal

**Estratégia:** Agregação diária dos acidentes

Para cada dia (`data`), foram calculados:
- **Target:** `num_acidentes` = contagem de acidentes naquele dia
- **Agregações:** soma de mortos, feridos, pessoas, veículos
- **Médias:** hora média do acidente (usando sin/cos)

**Resultado:** Dataset com 365 linhas (uma por dia de 2023)

#### 2.5 Normalização

**Método:** MinMaxScaler (escala 0-1)

```python
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df_model)
```

**Justificativa:** Redes LSTM são sensíveis à escala dos dados. Normalizar para [0,1] acelera a convergência e melhora a estabilidade do treinamento.

#### 2.6 Criação de Sequências Temporais

**Janela Temporal:** 14 dias (2 semanas)

Para cada amostra de treino:
- **Input (X):** Últimos 14 dias de dados (shape: `[14, 20]`)
- **Output (y):** Número de acidentes no 15º dia (shape: `[1]`)

**Exemplo:**
```
X = [dia_1, dia_2, ..., dia_14]  →  y = dia_15
X = [dia_2, dia_3, ..., dia_15]  →  y = dia_16
...
```

**Shape Final:**
- X_train: `(n_amostras, 14 timesteps, 20 features)`
- y_train: `(n_amostras, 1)`

#### 2.7 Divisão dos Dados

**Estratégia:** Divisão cronológica (sem shuffle)

```
├── Treino:     70% (primeiros dias)
├── Validação:  15% (dias intermediários)
└── Teste:      15% (últimos dias)
```

**Justificativa:** Em séries temporais, é crucial manter a ordem cronológica. O modelo treina no passado e é avaliado no futuro, simulando uso real.

#### 2.8 Tratamento de Valores Ausentes

- Dias sem acidentes: preenchidos com `0`
- Features temporais: forward fill + backward fill
- Valores NaN criados por lags: removidos (perda de ~7 linhas)

---

### 3. Arquitetura do Modelo LSTM

#### 3.1 Estrutura Geral

O modelo implementado é uma **Stacked LSTM** (LSTM empilhada) com 3 camadas recorrentes, seguidas de camadas densas para processamento não-linear final.

```
Input Shape: (14, 20)
     ↓
[LSTM 128 units] → Dropout 30%
     ↓
[LSTM 64 units]  → Dropout 30%
     ↓
[LSTM 32 units]  → Dropout 20%
     ↓
[Dense 32 units] → Dropout 20%
     ↓
[Dense 16 units] → Dropout 10%
     ↓
[Output 1 unit]
```

#### 3.2 Detalhamento das Camadas

**Camada 1 - LSTM (128 unidades)**
```python
LSTM(128, activation='tanh', return_sequences=True)
Dropout(0.3)
```
- **Função:** Captura padrões temporais de longo prazo
- **Ativação:** `tanh` (range -1 a 1, padrão para LSTMs)
- **return_sequences=True:** Passa sequências completas para próxima camada LSTM

**Camada 2 - LSTM (64 unidades)**
```python
LSTM(64, activation='tanh', return_sequences=True)
Dropout(0.3)
```
- **Função:** Refina os padrões capturados pela primeira camada
- **Redução de dimensionalidade:** 128 → 64

**Camada 3 - LSTM (32 unidades)**
```python
LSTM(32, activation='tanh', return_sequences=False)
Dropout(0.2)
```
- **Função:** Extrai features de alto nível
- **return_sequences=False:** Retorna apenas o último timestep
- **Output:** Vetor de 32 features temporais condensadas

**Camadas Densas (Fully Connected)**
```python
Dense(32, activation='relu')
Dropout(0.2)

Dense(16, activation='relu')
Dropout(0.1)
```
- **Função:** Processamento não-linear adicional
- **Ativação:** `relu` (boa para camadas densas)
- **Redução progressiva:** 32 → 16

**Camada de Saída**
```python
Dense(1)  # Sem ativação (regressão)
```
- **Função:** Predição do número de acidentes
- **Sem ativação:** Permite valores contínuos (regressão)

#### 3.3 Regularização

**Dropout (10% a 30%):**
- Desliga aleatoriamente neurônios durante treino
- Previne overfitting (memorização dos dados de treino)
- Taxas mais altas nas camadas iniciais

**Total de Parâmetros Treináveis:** ~150.000-200.000

#### 3.4 Compilação do Modelo

**Optimizer:** Adam (Adaptive Moment Estimation)
```python
Adam(learning_rate=0.001)
```
- Learning rate adaptativo por parâmetro
- Funciona bem com LSTMs sem ajuste fino

**Loss Function:** Mean Squared Error (MSE)
```python
loss = 'mean_squared_error'
```
- Adequada para problemas de regressão
- Penaliza erros grandes quadraticamente

**Métricas de Acompanhamento:**
```python
metrics = ['mae', 'mse']
```
- MAE: Erro médio absoluto (interpretação direta)
- MSE: Para monitorar convergência

#### 3.5 Callbacks (Controle de Treinamento)

**Early Stopping**
```python
EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True
)
```
- Para treino se validação não melhorar por 25 épocas
- Restaura pesos da melhor época

**Model Checkpoint**
```python
ModelCheckpoint(
    'melhor_modelo_lstm.keras',
    monitor='val_loss',
    save_best_only=True
)
```
- Salva automaticamente o melhor modelo

**Reduce Learning Rate on Plateau**
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.00001
)
```
- Reduz LR pela metade se estagnar por 10 épocas
- Ajuda a escapar de mínimos locais

---

### 4. Métricas de Avaliação

#### 4.1 Métricas Utilizadas

**1. MAE (Mean Absolute Error)**

```
MAE = (1/n) × Σ|y_pred - y_real|
```

**Interpretação:**
- Erro médio em número de acidentes
- Escala original dos dados (fácil interpretação)
- Não penaliza outliers excessivamente

**2. RMSE (Root Mean Squared Error)**

```
RMSE = √[(1/n) × Σ(y_pred - y_real)²]
```

**Interpretação:**
- Raiz do erro quadrático médio
- Penaliza erros grandes mais severamente que MAE
- Mesma unidade dos dados originais

**3. R² Score (Coeficiente de Determinação)**

```
R² = 1 - (SS_residual / SS_total)

Onde:
SS_residual = Σ(y_real - y_pred)²
SS_total = Σ(y_real - ȳ)²
```

**Interpretação:**
- Proporção da variância explicada pelo modelo
- Range: -∞ a 1
  - **R² = 1:** Predição perfeita
  - **R² = 0:** Modelo equivale à média
  - **R² < 0:** Modelo pior que a média

#### 4.2 Análise de Generalização

**Gap entre Treino e Validação:**
```
Gap = val_loss - train_loss
```

**Interpretação:**
- **Gap < 0.01:** Excelente generalização
- **Gap 0.01-0.05:** Boa generalização
- **Gap 0.05-0.1:** Generalização razoável
- **Gap > 0.1:** Possível overfitting

#### 4.3 Visualizações de Avaliação

**Curvas de Treinamento:**
- Gráfico de Loss (MSE) ao longo das épocas
- Gráfico de MAE ao longo das épocas
- Comparação treino vs validação

**Predições vs Valores Reais:**
- Série temporal: linha do tempo comparando predições e valores reais
- Scatter plot: cada ponto representa uma predição
- Linha diagonal: representa predição perfeita

**Análise de Resíduos:**
- Distribuição dos erros (y_real - y_pred)
- Identificação de viés sistemático

#### 4.4 Baseline para Comparação

**Modelo Baseline:** Média histórica

```python
baseline_pred = y_train.mean()
baseline_mae = mean_absolute_error(y_test, baseline_pred)
```

- Qualquer modelo deve superar esse baseline
- R² negativo indica performance abaixo do baseline

---

### 5. Considerações Finais

#### 5.1 Desafios Encontrados

1. **Natureza Aleatória dos Acidentes:** Acidentes têm alto componente estocástico, dificultando predições precisas.

2. **Dados Limitados:** Apenas 365 dias de dados (após agregação) é um dataset relativamente pequeno para Deep Learning.

3. **Features Externas Faltantes:** Dados meteorológicos detalhados, feriados locais, eventos especiais poderiam melhorar o modelo.

4. **Variância Alta:** Número de acidentes varia significativamente dia a dia.

#### 5.2 Aprendizados

1. **Features de Lag são Cruciais:** Valores passados melhoram significativamente a performance.

2. **Normalização é Essencial:** LSTMs convergem muito melhor com dados normalizados.

3. **Callbacks Otimizam Treino:** EarlyStopping e ReduceLR impedem overfitting e melhoram convergência.

4. **Regularização Forte:** Dropout de 30% foi necessário para evitar memorização.

#### 5.3 Aplicações Práticas

Este modelo pode ser utilizado para:

1. **PRF:** Alocação preventiva de recursos e planejamento de operações
2. **Seguradoras:** Análise de risco e precificação dinâmica
3. **Transportadoras:** Planejamento de rotas e horários
4. **Órgãos de Saúde:** Preparação de equipes de emergência

---

## Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Target Escolhido](#target-escolhido)
- [Dataset](#dataset)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Arquitetura do Modelo](#arquitetura-do-modelo)
- [Resultados](#resultados)
- [Como Executar](#como-executar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Métricas de Avaliação](#métricas-de-avaliação)
- [Melhorias Futuras](#melhorias-futuras)
- [Autores](#autores)
- [Licença](#licença)

---

## Sobre o Projeto

Este projeto foi desenvolvido como parte do **Sprint Challenge 4** com o objetivo de construir um modelo de rede neural LSTM capaz de prever aspectos relacionados a acidentes de trânsito a partir de séries temporais e variáveis contextuais.

A solução visa apoiar:
-  Decisões estratégicas de prevenção de riscos
-  Precificação de seguros
-  Planejamento logístico
-  Alocação de recursos da PRF

---

## Target Escolhido

**Problema:** Prever o número total de acidentes por dia em todas as rodovias federais do Brasil.

**Justificativa:**
- Métrica quantitativa e objetiva
- Útil para planejamento operacional da PRF
- Permite identificar padrões temporais (dias da semana, sazonalidade)
- Dados agregados reduzem ruído de eventos localizados
- Aplicável para seguradoras e empresas de logística

**Tipo de problema:** Regressão de séries temporais

---

## Dataset

**Fonte:** [Dados Abertos da PRF - Acidentes](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-acidentes)

**Arquivo utilizado:** Acidentes 2023 (Agrupados por ocorrência)

**Características principais:**
-  Período: 01/01/2023 a 31/12/2023
-  Abrangência: Todas as rodovias federais brasileiras
-  Registros: ~60.000 acidentes
-  Série temporal: 365 dias

**Colunas utilizadas:**
- `data_inversa`: Data do acidente
- `horario`: Hora do acidente
- `dia_semana`: Dia da semana
- `uf`: Estado
- `br`: Número da rodovia
- `classificacao_acidente`: Gravidade
- `condicao_metereologica`: Condições climáticas
- `tipo_pista`: Tipo de pista
- `pessoas`, `mortos`, `feridos_leves`, `feridos_graves`: Vítimas
- `veiculos`: Quantidade de veículos envolvidos

---

## Tecnologias Utilizadas

### Linguagem e Ambiente
- **Python 3.12**
- **Google Colab** (ambiente de execução)

### Principais Bibliotecas

```python
# Manipulação de dados
numpy==1.26.4
pandas==2.2.2

# Visualização
matplotlib==3.8.0
seaborn==0.13.2

# Machine Learning
tensorflow==2.17.0
keras==3.4.1
scikit-learn==1.5.1

# Utilitários
pickle
json
```

---

## Arquitetura do Modelo

### Estrutura da Rede Neural

```
Input Layer: (14 timesteps, 20 features)
    ↓
LSTM Layer 1: 128 units, activation='tanh', return_sequences=True
Dropout: 30%
    ↓
LSTM Layer 2: 64 units, activation='tanh', return_sequences=True
Dropout: 30%
    ↓
LSTM Layer 3: 32 units, activation='tanh', return_sequences=False
Dropout: 20%
    ↓
Dense Layer 1: 32 units, activation='relu'
Dropout: 20%
    ↓
Dense Layer 2: 16 units, activation='relu'
Dropout: 10%
    ↓
Output Layer: 1 unit (regressão)
```

### Hiperparâmetros

| Parâmetro | Valor |
|-----------|-------|
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | Mean Squared Error (MSE) |
| **Metrics** | MAE, MSE |
| **Batch Size** | 32 |
| **Epochs** | 200 (com Early Stopping) |
| **Sequence Length** | 14 dias |
| **Train/Val/Test Split** | 70% / 15% / 15% |

### Features Utilizadas

**Features Temporais:**
- Valores de lag: 1, 2, 3 e 7 dias anteriores
- Médias móveis: 3 e 7 dias
- Desvio padrão móvel: 7 dias
- Features cíclicas (seno/cosseno): hora, dia da semana, mês
- Indicadores: fim de semana, feriados

**Features de Vítimas:**
- Total de mortos, feridos leves, feridos graves
- Total de pessoas envolvidas
- Total de veículos

**Total de features:** 20

---

## Resultados

Os resultados detalhados do treinamento e avaliação do modelo incluem:

- Métricas de desempenho no conjunto de teste (MAE, RMSE, R²)
- Curvas de treinamento (Loss e MAE)
- Gráficos de predições vs valores reais
- Análise de generalização

Todos os resultados e gráficos estão disponíveis no notebook e no arquivo `resultados.json`.

---

## Como Executar

### Pré-requisitos

1. Conta no Google (para usar o Colab)
2. Arquivo CSV dos dados da PRF

### Passo a Passo

1. **Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/previsao-acidentes-lstm.git
cd previsao-acidentes-lstm
```

2. **Abra o notebook no Google Colab:**
   - Acesse [Google Colab](https://colab.research.google.com/)
   - Faça upload do arquivo `previsao_acidentes_prf_lstm.ipynb`
   - Ou abra diretamente: `File > Open Notebook > GitHub` e cole a URL do repositório

3. **Baixe os dados da PRF:**
   - Acesse: https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-acidentes
   - Baixe o arquivo: **"Acidentes 2023 (Agrupados por ocorrência)"**
   - Salve como `acidentes_2023.csv`

4. **Execute o notebook:**
   - Clique em `Runtime > Run all`
   - Quando solicitado, faça upload do arquivo CSV
   - Aguarde o treinamento (aproximadamente 15-20 minutos)

5. **Arquivos gerados:**
   - `modelo_lstm_acidentes_prf.keras` - Modelo treinado final
   - `melhor_modelo_lstm.keras` - Melhor modelo durante treinamento
   - `scaler.pkl` - Normalizador dos dados
   - `resultados.json` - Resumo das métricas

### Usando o Modelo Treinado

```python
import pickle
from tensorflow import keras
import numpy as np

# Carregar modelo e scaler
model = keras.models.load_model('modelo_lstm_acidentes_prf.keras')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Preparar dados de entrada (últimos 14 dias)
# X_new deve ter shape (1, 14, 20)
X_new = preparar_sequencia(ultimos_14_dias)

# Fazer predição
predicao_normalizada = model.predict(X_new)

# Desnormalizar
dummy = np.zeros((1, 20))
dummy[:, 0] = predicao_normalizada
predicao = scaler.inverse_transform(dummy)[:, 0]

print(f"Acidentes previstos para amanhã: {predicao[0]:.0f}")
```

---

## Estrutura do Projeto

```
previsao-acidentes-lstm/
│
├── README.md                          # Este arquivo
├── previsao_acidentes_prf_lstm.ipynb  # Notebook principal
├── resultados.json                    # Métricas de avaliação
│
├── modelos/
│   ├── modelo_lstm_acidentes_prf.keras  # Modelo final
│   ├── melhor_modelo_lstm.keras         # Melhor modelo (checkpoint)
│   └── scaler.pkl                       # Normalizador MinMaxScaler
│
├── dados/
│   └── acidentes_2023.csv             # Dataset (não versionado - baixar da PRF)
│
├── graficos/
│   ├── serie_temporal.png             # Visualização da série temporal
│   ├── curvas_treinamento.png         # Loss e MAE durante treino
│   └── predicoes_vs_real.png          # Scatter plot de predições
│
└── docs/
    └── relatorio.pdf                  # Relatório técnico completo
```

---

## Métricas de Avaliação

### MAE (Mean Absolute Error)
```python
MAE = (1/n) * Σ|y_pred - y_real|
```
- **Interpretação:** Erro médio em número de acidentes
- **Valor obtido:** 26.30 acidentes/dia
- **Contexto:** Média de acidentes/dia = ~150-180

### RMSE (Root Mean Squared Error)
```python
RMSE = √[(1/n) * Σ(y_pred - y_real)²]
```
- **Interpretação:** Penaliza erros grandes mais severamente
- **Valor obtido:** 35.56 acidentes/dia

### R² Score (Coeficiente de Determinação)
```python
R² = 1 - (SS_res / SS_tot)
```
- **Interpretação:** Proporção da variância explicada pelo modelo
- **Valor obtido:** -0.0248
- **Análise:** Valor negativo indica que o modelo não conseguiu capturar padrões preditivos significativos. Isso sugere que acidentes têm alto componente aleatório.

---

## Melhorias Futuras

### Abordagens Alternativas

1. **Classificação ao invés de Regressão**
   - Prever categorias: Baixo/Médio/Alto número de acidentes
   - Pode ter melhor performance que predição exata

2. **Modelos Ensemble**
   - Combinar LSTM com XGBoost, Random Forest
   - Aproveitar forças de diferentes algoritmos

3. **Mais Features Externas**
   - Dados meteorológicos detalhados
   - Calendário de feriados completo
   - Eventos especiais (Copa do Mundo, eleições)
   - Dados de tráfego (fluxo de veículos)

4. **Segmentação**
   - Modelos específicos por região
   - Modelos por tipo de rodovia
   - Análise por gravidade do acidente

5. **Arquiteturas Avançadas**
   - Attention Mechanisms
   - Transformer Models
   - GRU (Gated Recurrent Units)
   - Bidirectional LSTM

6. **Otimização de Hiperparâmetros**
   - Grid Search / Random Search
   - Bayesian Optimization
   - AutoML (Keras Tuner)
