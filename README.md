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

- Variáveis usadas: `Pclass`, `Sex`, `Age`, `Fare`
- Tratamento: idade faltante preenchida com a mediana; `Sex` codificado
  em 0/1
- Divisão treino/teste: 80/20 (`random_state=42`)
- Modelo: Regressão Logística (`max_iter=1000`)

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

## Principais aprendizados

- **Baselines importam.** No split principal, o modelo superou a regra
  simples por apenas 4 passageiros (144 contra 140 acertos).
- **Um único split engana.** A mesma configuração oscilou entre 72,1% e
  84,4% conforme o sorteio. Diferenças de ~2 pontos ficam dentro do
  ruído, o que motiva o uso de validação cruzada.
- **Desbalanceamento de classes.** O recall é maior para "Morreu"
  (0,86) do que para "Sobreviveu" (0,73): o modelo tende a favorecer a
  classe majoritária.
- **Acurácia não conta a história toda.** A matriz de confusão mostra
  *onde* o modelo erra, e não só quanto.

## Roadmap

- [x] Análise exploratória
- [x] Baselines e modelo de Regressão Logística
- [x] Matriz de confusão, precisão e recall
- [x] Análise da variação por `random_state`
- [ ] Validação cruzada (k-fold estratificado)
- [ ] Comparar Árvore de Decisão e Random Forest
- [ ] Feature engineering (tamanho da família, título, deck)
- [ ] Pipeline do scikit-learn (evitar data leakage)
- [ ] Submissão no Kaggle

## Como executar

1. Abra `titanic.ipynb` no [Google Colab](https://colab.research.google.com)
2. Execute as células em ordem

## Tecnologias

Python · pandas · matplotlib · seaborn · scikit-learn