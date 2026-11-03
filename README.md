# 🚢 Titanic: Previsão de Sobrevivência

Projeto de Machine Learning para prever quais passageiros do Titanic
sobreviveram, usando Python e scikit-learn. É meu primeiro projeto
prático de ML, feito para aprender o fluxo completo partindo do zero,
com foco em **entender cada etapa** e avaliar os modelos com rigor, e
não apenas em maximizar a acurácia.

## Objetivo

Construir um classificador que, dado o perfil do passageiro (classe,
sexo, idade, tarifa, família), preveja se ele sobreviveu, e verificar
se os modelos realmente superam regras simples.

## Dataset

Dados do [Titanic no Kaggle](https://www.kaggle.com/c/titanic):
891 passageiros, com variáveis demográficas e da viagem.
Alvo: `Survived` (0 = morreu, 1 = sobreviveu).

## Abordagem

- **Variáveis base:** `Pclass`, `Sex`, `Age`, `Fare`
- **Variáveis de família:** `SibSp`, `Parch`, `FamilySize`, `IsAlone` e `FamilyGroup` (faixas: sozinho, 2 a 4 pessoas, 5 ou mais), esta última também em *one-hot encoding*
- **Tratamento:** idade faltante preenchida com a mediana; `Sex` codificado em 0/1
- **Avaliação:** split 80/20, validação cruzada estratificada (5 folds), validação cruzada repetida (5 folds × 5 a 10 repetições) e 100 splits aleatórios para medir a instabilidade
- **Modelos:** Regressão Logística, Árvore de Decisão e Random Forest
- **Referências de comparação (baselines):** "sempre prever morreu" e a regra "mulher sobrevive, homem morre"

## Resultados

### 1. Baselines e primeiro modelo (split único, `random_state=42`)

| Abordagem | Acurácia no teste |
|---|---|
| Sempre prever "morreu" | 58,65% |
| Regra: "mulher sobrevive, homem morre" | 78,21% |
| Regressão Logística | 80,44% |

### 2. Matriz de confusão, precisão e recall (Regressão Logística)

| | Previu "Morreu" | Previu "Sobreviveu" |
|---|---|---|
| **Morreu** (105) | 90 | 15 |
| **Sobreviveu** (74) | 20 | 54 |

| Classe | Precisão | Recall | F1 |
|---|---|---|---|
| Morreu | 0,82 | 0,86 | 0,84 |
| Sobreviveu | 0,78 | 0,73 | 0,76 |

### 3. Instabilidade do split único (100 splits, Regressão Logística)

| Média | Desvio padrão | Mínimo | Máximo |
|---|---|---|---|
| 78,9% | 3,0 p.p. | 72,1% | 84,4% |

### 4. Comparação de modelos (validação cruzada, 5 folds, variáveis base)

| Modelo | Média | Desvio |
|---|---|---|
| Regressão Logística | 0,789 | 0,016 |
| Árvore de Decisão (`max_depth=5`) | 0,819 | 0,013 |
| Random Forest (100 árvores) | 0,833 | 0,033 |

> Nota: com validação cruzada **repetida** (mais estável), a Regressão Logística
> fica em 0,788 ± 0,032 e a Random Forest em **0,824 ± 0,023**. O 0,833 da tabela
> acima foi favorecido pelo sorteio de um único conjunto de 5 folds.

### 5. Modelos contra a regra simples (100 splits, variáveis base)

| Modelo | Diferença média vs. regra | Vitórias em 100 splits |
|---|---|---|
| Regressão Logística | +0,1 p.p. | 50 |
| Random Forest | +3,3 p.p. | 83 |

### 6. Overfitting da Árvore de Decisão (split `random_state=42`)

| `max_depth` | Treino | Teste |
|---|---|---|
| 1 | 0,788 | 0,782 |
| 2 | 0,803 | 0,765 |
| 3 | 0,827 | 0,799 |
| 5 | 0,857 | 0,804 |
| 10 | 0,917 | 0,782 |
| Sem limite | 0,978 | 0,732 |

### 7. Importância das variáveis (Random Forest, variáveis base)

| Variável | Importância por impureza (treino) | Permutation importance (teste) |
|---|---|---|
| Sex | 0,285 | 0,190 |
| Pclass | 0,100 | 0,094 |
| Age | 0,287 | 0,026 |
| Fare | 0,328 | 0,015 |

### 8. Feature engineering: família

**8.1 Exploração.** A sobrevivência por tamanho da família forma uma curva, e não uma reta:

| Tamanho da família | Taxa de sobrevivência | Passageiros |
|---|---|---|
| 1 (sozinho) | 30,4% | 537 |
| 2 | 55,3% | 161 |
| 3 | 57,8% | 102 |
| 4 | 72,4% | 29 |
| 5 | 20,0% | 15 |
| 6 | 13,6% | 22 |
| 7 | 33,3% | 12 |
| 8 | 0,0% | 6 |
| 11 | 0,0% | 7 |

**8.2 Primeira tentativa (faixas codificadas como 0/1/2, validação repetida 5×5).**

| Variáveis | Logística | Random Forest |
|---|---|---|
| Base | 0,788 ± 0,032 | 0,824 ± 0,023 |
| + `FamilySize` | 0,791 ± 0,030 | 0,820 ± 0,023 |
| + `IsAlone` | 0,784 ± 0,033 | 0,822 ± 0,022 |
| + `FamilyGroup` (0/1/2) | 0,785 ± 0,032 | 0,823 ± 0,020 |

Nenhuma variante ajudou. Para a Regressão Logística, a codificação 0/1/2 é lida como
escala linear e não representa o padrão real (baixo, alto, baixo).

**8.3 Correção: `FamilyGroup` em one-hot (validação repetida 5×10, comparação pareada fold a fold).**

| Modelo | Base | + one-hot | Diferença média | One-hot vence em |
|---|---|---|---|---|
| Regressão Logística | 0,788 ± 0,032 | 0,812 ± 0,027 | +0,025 | 44 de 50 folds |
| Random Forest | 0,824 ± 0,023 | 0,824 ± 0,020 | -0,001 | 21 de 50 folds |

Com a codificação adequada, a família melhora de forma consistente a Regressão Logística
(cerca de 22 passageiros a mais acertados em 891) e a distância para a Random Forest
cai de 3,6 para 1,2 p.p. A Random Forest não muda, pois já captura padrões não lineares.

### 9. Ajuste de `max_depth` com GridSearchCV (Árvore, com `SibSp`, `Parch` e `FamilySize`)

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
- O `max_depth=5` da seção 4 foi escolhido olhando um único split de teste; o GridSearchCV (seção 9) refaz essa escolha da forma correta.
- Os folds de repetições diferentes reutilizam os mesmos passageiros, então a contagem de vitórias entre folds é um guia, e não um teste estatístico formal.
- O melhor resultado do GridSearchCV é levemente otimista (sem validação cruzada aninhada).
- A Random Forest foi usada com hiperparâmetros padrão (sem ajuste).

## Roadmap

- [x] Análise exploratória
- [x] Baselines e Regressão Logística
- [x] Matriz de confusão, precisão e recall
- [x] Análise da variação por `random_state`
- [x] Validação cruzada (k-fold estratificado e repetido)
- [x] Árvore de Decisão e Random Forest
- [x] Permutation importance
- [x] Feature engineering de família (`FamilySize`, `IsAlone`, `FamilyGroup`, one-hot)
- [x] GridSearchCV
- [ ] Extrair o título do nome (Mr, Mrs, Miss, Master)
- [ ] Pipeline do scikit-learn (evitar data leakage)
- [ ] Ajuste de hiperparâmetros da Random Forest
- [ ] Submissão no Kaggle

## Como executar

1. Abra `titanic.ipynb` no [Google Colab](https://colab.research.google.com)
2. Execute as células em ordem

## Tecnologias

Python · pandas · numpy · matplotlib · seaborn · scikit-learn