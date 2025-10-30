# Previsão de Acidentes com LSTMs - Sprint Challenge 4

Projeto de Deep Learning utilizando redes neurais recorrentes (LSTM) para prever padrões de acidentes em rodovias brasileiras com base nos dados públicos da Polícia Rodoviária Federal (PRF).

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
- Decisões estratégicas de prevenção de riscos
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
