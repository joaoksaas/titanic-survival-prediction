# 🚢 Titanic: Previsão de Sobrevivência

Projeto de Machine Learning para prever quais passageiros do Titanic
sobreviveram, usando Python e scikit-learn. Feito para aprender o fluxo completo partindo do zero,
com foco em **entender cada etapa** e avaliar os modelos com rigor, e
não apenas em maximizar a acurácia.

## Resumo

- **Objetivo:** prever `Survived` (0 = morreu, 1 = sobreviveu) a partir do perfil do passageiro.
- **Método:** baselines simples, validação cruzada (inclusive repetida e aninhada), comparações pareadas fold a fold, Pipeline do scikit-learn e interpretação dos modelos.
- **Resultado principal:** os modelos chegam a ~83% em validação cruzada, apenas ~4 a 5 pontos acima da regra "mulher sobrevive, homem morre". No teste do Kaggle, ficaram muito próximos dessa regra.

| Etapa | Acurácia |
|---|---|
| Regra "mulher sobrevive" (split único) | 78,21% |
| Regressão Logística, 4 variáveis (validação cruzada repetida) | 78,8% |
| + família (one-hot) | 81,2% |
| + título e família | 82,3% |
| Pipeline (sem vazamento de dados) | 82,7% |
| Random Forest ajustada (validação aninhada) | 83,4% |
| + `Embarked`, `Deck`, `TicketGroup` | sem ganho |

> Os protocolos de validação variam um pouco entre as linhas; vale a ordem de grandeza, e não o decimal.

| Submissão no Kaggle | Nota |
|---|---|
| Regressão Logística (Pipeline, `Title` + `FamilyGroup`) | 0,77033 |
| Random Forest ajustada (`min_samples_leaf=5`) | 0,75358 |
| Referência: regra "só mulheres sobrevivem" (nota pública) | 0,76555 |

## Objetivo

Construir um classificador que, dado o perfil do passageiro (classe,
sexo, idade, tarifa, família, título), preveja se ele sobreviveu, e
verificar se os modelos realmente superam regras simples.

## Dataset

Dados do [Titanic no Kaggle](https://www.kaggle.com/c/titanic):
891 passageiros no treino e 418 no teste.
Alvo: `Survived` (0 = morreu, 1 = sobreviveu).

## Abordagem

- **Variáveis base:** `Pclass`, `Sex`, `Age`, `Fare`
- **Variáveis de família:** `SibSp`, `Parch`, `FamilySize`, `IsAlone` e `FamilyGroup` (faixas: sozinho, 2 a 4 pessoas, 5 ou mais), esta última também em *one-hot encoding*
- **Título:** extraído de `Name` (Mr, Miss, Mrs, Master e Rare, que agrupa os títulos pouco frequentes), em *one-hot encoding*
- **Testadas sem ganho:** `Embarked`, `Deck` (primeira letra de `Cabin`) e `TicketGroup` (quantos passageiros dividem o bilhete)
- **Tratamento:** Pipeline do scikit-learn com imputação da mediana, padronização das numéricas e one-hot das categóricas, tudo aprendido só com o treino em cada rodada
- **Avaliação:** split 80/20, validação cruzada estratificada, validação cruzada repetida, 100 splits aleatórios para medir a instabilidade, validação aninhada para o ajuste de hiperparâmetros; comparações entre variantes feitas fold a fold (pareadas)
- **Modelos:** Regressão Logística, Árvore de Decisão e Random Forest
- **Referências de comparação (baselines):** "sempre prever morreu" e a regra "mulher sobrevive, homem morre"

## Resultados detalhados

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

### 9. Feature engineering: título extraído do nome

**9.1 Exploração.**

| Título | Taxa de sobrevivência | Passageiros | Sem idade registrada |
|---|---|---|---|
| Mrs | 79,4% | 126 | 17 |
| Miss | 70,3% | 185 | 36 |
| Master | 57,5% | 40 | 4 |
| Rare | 34,8% | 23 | 1 |
| Mr | 15,7% | 517 | 119 |

Dos 177 passageiros sem idade, 119 (67%) são "Mr" e só 4 são "Master": o título quase
não recupera a informação de idade onde ela falta.

**9.2 Experimento pareado (validação repetida 5×10, diferença contra a base).**

| Modelo | Variante | Média | Diferença vs. base | Vence em |
|---|---|---|---|---|
| Regressão Logística | Base | 0,788 | | |
| | + Title | 0,793 | +0,005 | 29 de 50 |
| | + Família | 0,812 | +0,025 | 44 de 50 |
| | + Title + Família | 0,823 | +0,035 | 45 de 50 |
| Random Forest | Base | 0,824 | | |
| | + Title | 0,824 | 0,000 | 25 de 50 |
| | + Família | 0,824 | -0,001 | 21 de 50 |
| | + Title + Família | 0,821 | -0,004 | 19 de 50 |

**9.3 Título sobre a família (Regressão Logística, comparação direta).**
Title + Família contra Família sozinha: **+0,011** em média, com vitórias em 33 de 50 folds.
Um ganho pequeno, acima do acaso, mas bem menos consistente que o da família sobre a base.

### 10. Ajuste de `max_depth` com GridSearchCV (Árvore, com `SibSp`, `Parch` e `FamilySize`)

| `max_depth` | Média | Desvio |
|---|---|---|
| 2 | 0,777 | 0,015 |
| 3 | 0,823 | 0,015 |
| 4 | 0,819 | 0,012 |
| 5 | 0,811 | 0,017 |
| 6 | 0,818 | 0,025 |
| 8 | 0,822 | 0,016 |
| Sem limite | 0,772 | 0,024 |

### 11. Pipeline e interpretação da Regressão Logística

**11.1 Acurácia com o Pipeline (validação repetida 5×10).**

| Modelo | Antes (Title + Família) | Pipeline |
|---|---|---|
| Regressão Logística | 0,823 | 0,827 ± 0,024 |
| Random Forest | 0,821 | 0,819 ± 0,022 |

As diferenças (de 0,2 a 0,4 p.p.) estão dentro do ruído: corrigir o vazamento de dados custou, na prática, nada mensurável.

**11.2 O que a Regressão Logística aprendeu** (variáveis numéricas padronizadas; categorias lidas por diferenças dentro de cada grupo).

| Variável | Efeito |
|---|---|
| `Pclass` | -0,81 por desvio padrão (≈ -0,96 por classe; odds ×0,38) |
| `Age` | -0,30 por desvio padrão (≈ -0,023 por ano) |
| `Fare` | +0,21 por desvio padrão (≈ +0,004 por libra) |
| Sexo | mulher contra homem: +1,81 (odds ×6,1) |
| Título (contra Master) | Mrs -1,36 · Miss -1,88 · Rare -2,62 · Mr -2,98 |
| Família (contra sozinho) | 2 a 4: -0,22 (odds ×0,80) · 5 ou mais: -2,61 (odds ×0,07) |

O que o modelo usa da família é sobretudo a penalidade das famílias grandes (~62
passageiros); a vantagem de famílias de 2 a 4 sobre viajar sozinho praticamente
desaparece depois de controlar sexo, classe e título.

### 12. Ajuste de hiperparâmetros com validação aninhada (15 rodadas externas, 5 folds internos)

| Modelo | Padrão | Ajustado | Diferença | Ajustado vence / empata / perde |
|---|---|---|---|---|
| Regressão Logística (`C`) | 0,827 | 0,828 | +0,001 | 2 / 13 / 0 |
| Random Forest (`max_depth`, `min_samples_leaf`) | 0,819 | 0,834 | +0,015 | 11 / 2 / 2 |

A Random Forest escolheu `max_depth=None` e `min_samples_leaf=5`. A nota do `GridSearchCV`
(0,850) é otimista, porque escolhe o melhor de 12 candidatos nos mesmos folds em que mede; a
validação aninhada (0,834) é a estimativa honesta. A Regressão Logística já estava perto do
ponto ótimo com os valores padrão.

### 13. Novas variáveis: `Embarked`, `Deck` e `TicketGroup` (validação repetida 5×10, pareada)

| Modelo | Variante | Média | Vence / empata / perde |
|---|---|---|---|
| Regressão Logística | Base | 0,827 | |
| | + Embarked | 0,827 | 15 / 20 / 15 |
| | + Deck | 0,827 | 20 / 11 / 19 |
| | + TicketGroup | 0,828 | 7 / 42 / 1 |
| | + as três | 0,827 | 23 / 6 / 21 |
| Random Forest | Base | 0,836 | |
| | + Embarked | 0,833 | 14 / 9 / 27 |
| | + Deck | 0,833 | 15 / 10 / 25 |
| | + TicketGroup | 0,833 | 10 / 16 / 24 |
| | + as três | 0,830 | 13 / 3 / 34 |

Nenhuma ajudou. O `Deck` é em boa parte outra forma de dizer "1ª classe" (86% dos passageiros
com deck conhecido são da 1ª classe), e o `TicketGroup` concorda com o `FamilyGroup` em 79% dos
passageiros. Na Random Forest, as colunas extras pioram levemente o resultado.

### 14. Submissão no Kaggle

| Modelo | Nota | Acertos (de 418) |
|---|---|---|
| Regressão Logística (Pipeline, `Title` + `FamilyGroup`) | 0,77033 | 322 |
| Random Forest ajustada | 0,75358 | 315 |
| Referência: regra "só mulheres sobrevivem" (nota pública) | 0,76555 | 320 |

Os dois modelos discordam em 23 dos 418 passageiros, e a diferença de 7 acertos entre eles vem
desses casos (a Regressão Logística acertou 15, a Random Forest 8). Previram, respectivamente,
39,7% e 37,6% de sobreviventes (no treino, 38%).

As notas ficaram abaixo da validação cruzada (~0,83). Contribuições possíveis, sem como separá-las:
ruído de um teste pequeno (erro padrão de ~2 pontos com 418 passageiros), um teste um pouco mais
difícil (a própria regra simples cai do treino para o teste) e otimismo de ter escolhido variáveis
e hiperparâmetros olhando os mesmos dados.

## Principais aprendizados

- **Baselines importam.** No split principal, a Regressão Logística superou a regra simples por apenas 4 passageiros (144 contra 140 acertos). No teste do Kaggle, a diferença foi de 2 passageiros para a logística e de -5 para a floresta.
- **Um único split engana.** A mesma configuração variou de 72,1% a 84,4% conforme o sorteio. A validação cruzada (de preferência repetida) é a base de comparação.
- **O modelo certo captura interações.** Com as mesmas variáveis, a Random Forest superou a regra em 83 de 100 splits, enquanto a Regressão Logística empatou.
- **Overfitting é visível.** Uma árvore sem limite acerta 97,8% no treino e só 73,2% no teste, pior que a regra simples.
- **Importância por impureza engana.** `Fare` e `Age` (contínuas, com muitos pontos de corte) aparecem inflacionadas; na permutation importance, `Sex` e `Pclass` dominam.
- **A codificação importa.** Categorias codificadas como números (0/1/2) são lidas como escala por modelos lineares. Em one-hot, a mesma informação passou a ajudar a Regressão Logística (+2,5 p.p.).
- **Feature engineering ajuda mais os modelos simples.** Família e título melhoraram a Regressão Logística (de 0,788 para 0,823) e não alteraram a Random Forest, que já captura padrões não lineares.
- **Redundância e tamanho do subgrupo limitam o ganho.** O título sozinho agregou pouco (repete `Sex` e `Age`), e `Deck` e `TicketGroup` repetem `Pclass` e `FamilyGroup`.
- **Modelos interpretáveis explicam o ganho.** Os pesos da logística mostram que o valor da família está na penalidade das famílias grandes, e não em viajar acompanhado.
- **Compare de forma pareada.** Comparar versões fold a fold, olhando vitórias, empates e derrotas, é mais sensível do que comparar médias com seus desvios.
- **Ajuste de hiperparâmetros ajuda quem está longe do ponto ótimo.** A floresta ganhou 1,5 ponto; a logística, nada. A nota do `GridSearchCV` é otimista, e a validação aninhada dá o número honesto.
- **A diferença entre dois modelos é decidida pelos casos em que eles discordam.** Com 23 discordâncias em 418, uma diferença de poucos acertos é ruído.
- **Resultados negativos também são resultados.** Saber que uma variável não ajuda evita complexidade inútil.
- **Desconfie de resultados idênticos.** Oito resultados iguais para modelos diferentes revelaram um erro no código, e não um achado.
- **Acurácia não conta tudo.** O recall é maior para "Morreu" (0,86) do que para "Sobreviveu" (0,73): o modelo favorece a classe majoritária.

## Limitações conhecidas

- A escolha dos títulos "raros", dos decks agrupados e dos hiperparâmetros foi feita olhando os dados de treino inteiros, o que deixa as estimativas de validação levemente otimistas.
- O `max_depth=5` da seção 4 foi escolhido olhando um único split de teste; o GridSearchCV (seção 10) refaz essa escolha da forma correta.
- Os folds de repetições diferentes reutilizam os mesmos passageiros, então a contagem de vitórias entre folds é um guia, e não um teste estatístico formal.
- Os pesos da Regressão Logística (seção 11.2) vêm de um modelo regularizado treinado em todos os dados; `Sex` e `Title` são parcialmente redundantes, então vale a leitura do conjunto.
- A nota do Kaggle se baseia em 418 passageiros, e diferenças de poucos acertos entre submissões não são conclusivas.

## Próximos passos

- Organizar o notebook como relatório (seções, índice e conclusão)
- Um segundo projeto, mais elaborado, com outro conjunto de dados

## Como executar

Requisitos: Python 3.10 ou superior (desenvolvido com 3.13).

1. Clone o repositório e, de preferência, crie um ambiente virtual.
2. Instale as bibliotecas:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Abra `titanic.ipynb` no VS Code (com a extensão Jupyter) ou no Jupyter Notebook e execute as células em ordem.
4. O conjunto de treino é baixado automaticamente de uma URL pública (é preciso ter internet).
5. Para a seção de submissão, baixe o `test.csv` em [kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data) e coloque-o na mesma pasta do notebook (o arquivo não é versionado).

## Tecnologias

Python · pandas · numpy · matplotlib · seaborn · scikit-learn