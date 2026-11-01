# 🚢 Titanic: Previsão de Sobrevivência

Projeto de Machine Learning para prever quais passageiros do Titanic
sobreviveram, usando Python e scikit-learn. É meu primeiro projeto
prático de ML, feito para aprender o fluxo completo partindo do zero,
com foco em **entender** cada etapa e não só em maximizar a acurácia.

## Objetivo

Construir um classificador que, dado o perfil do passageiro (classe,
sexo, idade, tarifa), preveja se ele sobreviveu, e avaliar com rigor
se o modelo realmente supera regras simples.

## Dataset

Dados do [Titanic no Kaggle](https://www.kaggle.com/c/titanic):
891 passageiros, com variáveis demográficas e da viagem.
Alvo: `Survived` (0 = morreu, 1 = sobreviveu).

## Abordagem atual

- **Variáveis (v1):** `Pclass`, `Sex`, `Age`, `Fare`
- **Variáveis (v2):** v1 + `SibSp`, `Parch`, `FamilySize`
- **Tratamento:** idade faltante preenchida com a mediana; `Sex` codificado em 0/1
- **Avaliação:** split 80/20, validação cruzada estratificada (5 folds) e 100 splits aleatórios para medir a instabilidade
- **Modelos:** Regressão Logística, Árvore de Decisão e Random Forest
- **Referências de comparação (baselines):** "sempre prever morreu" e a regra "mulher sobrevive, homem morre"


## Resultados

### Comparação com baselines (split único)

| Abordagem | Acurácia no teste |
|---|---|
| Baseline: sempre prever "morreu" | 58,65% |
| Regra simples: "mulher sobrevive, homem morre" | 78,21% |
| Regressão Logística | 80,44% |

### Matriz de confusão

| | Previu "Morreu" | Previu "Sobreviveu" |
|---|---|---|
| **Morreu** (105) | 90 | 15 |
| **Sobreviveu** (74) | 20 | 54 |

### Precisão e recall

| Classe | Precisão | Recall | F1 |
|---|---|---|---|
| Morreu | 0,82 | 0,86 | 0,84 |
| Sobreviveu | 0,78 | 0,73 | 0,76 |

### Estabilidade do resultado (100 splits aleatórios)

| Média | Desvio padrão | Mínimo | Máximo |
|---|---|---|---|
| 78,9% | 3,0 p.p. | 72,1% | 84,4% |

### Comparação de modelos (validação cruzada, 5 folds, variáveis v1)

| Modelo | Média | Desvio |
|---|---|---|
| Regressão Logística | 0,789 | 0,016 |
| Árvore de Decisão (`max_depth=5`) | 0,819 | 0,013 |
| Random Forest (100 árvores) | 0,833 | 0,033 |

### Modelos contra a regra simples (100 splits, variáveis v1)

| Modelo | Diferença média vs. regra | Vitórias em 100 splits |
|---|---|---|
| Regressão Logística | +0,1 p.p. | 50 |
| Random Forest | +3,3 p.p. | 83 |

### Overfitting da Árvore de Decisão (split `random_state=42`)

| `max_depth` | Treino | Teste |
|---|---|---|
| 1 | 0,788 | 0,782 |
| 2 | 0,803 | 0,765 |
| 3 | 0,827 | 0,799 |
| 5 | 0,857 | 0,804 |
| 10 | 0,917 | 0,782 |
| Sem limite | 0,978 | 0,732 |

### Importância das variáveis (Random Forest, v1)

| Variável | Importância por impureza (treino) | Permutation importance (teste) |
|---|---|---|
| Sex | 0,285 | 0,190 |
| Pclass | 0,100 | 0,094 |
| Age | 0,287 | 0,026 |
| Fare | 0,328 | 0,015 |

### Feature engineering: `FamilySize` (validação cruzada, variáveis v2)

| Modelo | v1 | v2 (com família) |
|---|---|---|
| Regressão Logística | 0,789 ± 0,016 | 0,788 ± 0,011 |
| Random Forest | 0,833 ± 0,033 | 0,816 ± 0,021 |

### Ajuste de `max_depth` com GridSearchCV (Árvore, variáveis v2)

| `max_depth` | Média | Desvio |
|---|---|---|
| 2 | 0,777 | 0,015 |
| 3 | 0,823 | 0,015 |
| 4 | 0,819 | 0,012 |
| 5 | 0,811 | 0,017 |
| 6 | 0,818 | 0,025 |
| 8 | 0,822 | 0,016 |
| Sem limite | 0,772 | 0,024 |

## Limitações conhecidas

- A mediana da idade foi calculada com o dataset inteiro antes da divisão, o que gera um vazamento leve de dados (*data leakage*). A correção prevista é usar Pipelines do scikit-learn.
- O `max_depth=5` da seção 4 foi escolhido olhando um único split de teste; o GridSearchCV refaz essa escolha da forma correta.
- Diferenças menores que um desvio padrão não são tratadas como conclusivas.
- O melhor resultado do GridSearchCV é levemente otimista (sem validação cruzada aninhada).

## Roadmap

- [x] Análise exploratória
- [x] Baselines e Regressão Logística
- [x] Matriz de confusão, precisão e recall
- [x] Análise da variação por `random_state`
- [x] Validação cruzada (k-fold estratificado)
- [x] Árvore de Decisão e Random Forest
- [x] Permutation importance
- [x] Primeira variável nova (`FamilySize`) e GridSearchCV
- [ ] Investigar `IsAlone` / faixas de tamanho de família
- [ ] Extrair o título do nome (Mr, Mrs, Miss, Master)
- [ ] Pipeline do scikit-learn (evitar data leakage)
- [ ] Ajuste de hiperparâmetros da Random Forest

## Como executar

1. Abra `titanic.ipynb` no [Google Colab](https://colab.research.google.com)
2. Execute as células em ordem

## Tecnologias

Python · pandas · numpy · matplotlib · seaborn · scikit-learn