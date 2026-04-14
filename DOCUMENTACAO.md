# Documentação Técnica — EDA COVID-19 com Apache Spark (PySpark Real)

**Disciplina:** Processamento de Grande Volume de Dados — UVV
**Dataset:** Our World in Data — COVID-19 (~570 mil registros, 61 colunas)
**Período coberto:** Janeiro/2020 → Fevereiro/2026
**Implementação:** Apache Spark real via PySpark 4.1.1 + Java 21

> Esta documentação cobre exclusivamente os arquivos que utilizam o **Apache Spark de verdade**:
> `src/main.py` e `notebook/main.ipynb`.
> Para a simulação em Python puro, consulte `DOCUMENTACAO_SIMULACAO.md`.

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Arquitetura e Fluxo de Dados](#2-arquitetura-e-fluxo-de-dados)
3. [Pipeline ETL — `src/main.py`](#3-pipeline-etl--srcmainpy)
4. [Notebook EDA — `notebook/main.ipynb`](#4-notebook-eda--notebookmainipynb)
5. [As 15 Visualizações](#5-as-15-visualizações)
6. [Conceitos de Big Data Aplicados](#6-conceitos-de-big-data-aplicados)
7. [Guia de Apresentação em Sala](#7-guia-de-apresentação-em-sala)

---

## 1. Visão Geral

O projeto realiza uma **Análise Exploratória de Dados (EDA)** sobre a pandemia de COVID-19 usando o **Apache Spark** (via PySpark) para processar um dataset de grande volume.

| Arquivo | Papel |
|---|---|
| `src/main.py` | Pipeline ETL automatizado — extrai, transforma e exporta dados |
| `notebook/main.ipynb` | Análise interativa com 15 visualizações |

**Por que Spark e não Pandas?**
Pandas carrega tudo em memória de uma única máquina. Spark processa em **partições distribuídas** — o mesmo código escala para bilhões de linhas em um cluster sem reescrever nada. Este projeto demonstra esse princípio com um dataset real.

---

## 2. Arquitetura e Fluxo de Dados

```mermaid
flowchart LR

    subgraph ENTRADA["ENTRADA"]
        A[("owid-covid.csv\n570 mil linhas\n61 colunas")]
    end

    subgraph SPARK["APACHE SPARK — PySpark 4.1.1"]
        direction TB
        B["SparkSession\nlocal[*]"]
        C["DataFrame Bruto\n(schema inferido)"]
        D["Filtragem\n• remove regiões agregadas\n• remove valores negativos"]
        E["Enriquecimento\n• year_month\n• daily_mortality_rate\n• fully_vaccinated_pct"]
        F["Window Function\n• média móvel 7 dias\n• particionada por país"]
        G["Agregações\n• por país\n• por continente\n• mensal global"]
    end

    subgraph SAIDA["SAÍDA"]
        H[("data/processado/\nresumao_por_pais\nevol_mensal\ncont")]
        I["notebook/main.ipynb\n15 visualizações"]
    end

    A --> B --> C --> D --> E --> F --> G
    G --> H
    G --> I
```

---

## 3. Pipeline ETL — `src/main.py`

O arquivo implementa as três fases clássicas de processamento de dados:

```mermaid
flowchart LR
    E["E — Extract\nLê o CSV bruto\ncom Spark"] --> T["T — Transform\nLimpa, filtra\ne enriquece"] --> L["L — Load\nSalva CSVs\nprocessados"]
```

---

### Módulo 1 — Configuração do Ambiente (Java + SparkSession)

```python
# Auto-detecção do JAVA_HOME a partir do JDK instalado via install-jdk
if not os.environ.get("JAVA_HOME"):
    jdk_base = os.path.expanduser("~/.jdk")
    ...
    os.environ["JAVA_HOME"] = os.path.join(jdk_base, entradas[0])
```

O PySpark é uma biblioteca Python que, por baixo dos panos, inicia uma **JVM (Java Virtual Machine)** para executar o Spark. Sem o Java configurado, o programa não consegue iniciar.

Usamos o pacote `install-jdk` para **embutir o Java como dependência Python**, eliminando a necessidade de instalar Java manualmente no sistema operacional.

```mermaid
sequenceDiagram
    participant P as Python (pip)
    participant J as install-jdk
    participant D as ~/.jdk/jdk-21/
    participant S as PySpark / JVM

    P->>J: pip install install-jdk
    P->>J: jdk.install('21')
    J->>D: baixa e extrai OpenJDK 21 (~200 MB)
    Note over D: salvo uma única vez
    P->>D: os.environ["JAVA_HOME"] = ~/.jdk/jdk-21/
    P->>S: SparkSession.builder.getOrCreate()
    S->>D: lê JAVA_HOME → inicia JVM
    S-->>P: SparkSession pronta
```

A `SparkSession` é o **ponto de entrada** de qualquer aplicação Spark. O padrão `builder.appName().master().getOrCreate()` cria ou reutiliza uma sessão existente.

---

### Módulo 2 — Extração (Extract)

```python
df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(caminho)
)
df = df.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))
```

O Spark lê o CSV de forma distribuída — internamente divide o arquivo em **partições** processadas em paralelo nos núcleos da CPU. `inferSchema` faz o Spark amostrar os dados para detectar tipos automaticamente.

**Resultado:** DataFrame com 570.606 linhas × 61 colunas, totalmente tipado.

---

### Módulo 3 — Análise de Qualidade dos Dados

```python
exprs_nulos = [
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in colunas_interesse
]
resultado = df.agg(*exprs_nulos).collect()[0].asDict()
```

Em vez de fazer uma query por coluna (61 queries), constrói **uma lista de expressões** e executa tudo em uma única passagem. Isso é possível pelo modelo de **Lazy Evaluation** do Spark.

**Principais achados:**

| Coluna | % Nulos | Motivo |
|---|---|---|
| `icu_patients` | 93% | Poucos países reportam UTI diariamente |
| `hosp_patients` | 93% | Idem — hospitalizações |
| `people_fully_vaccinated` | 87% | Vacinação começou apenas em Dez/2020 |
| `reproduction_rate` | 68% | Estimativa complexa, nem sempre calculada |
| `new_cases` | 3% | Registros antes do início da pandemia |

---

### Módulo 4 — Transformação (Transform)

```mermaid
flowchart TD
    A["DataFrame Bruto\n570.606 linhas"] --> B

    B{"Tem continente\npreenchido?"}
    B -- Não --> X1["Remove\n(região agregada:\nWorld, Asia, etc.)"]
    B -- Sim --> C

    C{"new_cases ou\nnew_deaths < 0?"}
    C -- Sim --> X2["Remove\n(erro de notificação)"]
    C -- Não --> D

    D["Adiciona colunas derivadas"]
    D --> E["year_month\n ex: '2021-03'"]
    D --> F2["daily_mortality_rate\n= mortes/casos × 100"]
    D --> G2["fully_vaccinated_pct\n= vacinados/população × 100"]
    D --> H2["new_cases_per_100k\n= casos/população × 100.000"]
```

O dataset OWID inclui registros como `"World"`, `"High income"`, `"European Union"` que são **somas de países** — mantê-los duplicaria os dados nas análises.

---

### Módulo 5 — Window Function (Média Móvel)

```python
window_spec = (
    Window
    .partitionBy("country")               # cada país é independente
    .orderBy(F.unix_date(F.col("date")))  # ordena por data
    .rowsBetween(-6, 0)                   # janela: 7 dias
)
df = df.withColumn("new_cases_ma7", F.avg("new_cases").over(window_spec))
```

A Window Function calcula um valor para cada linha levando em conta **linhas vizinhas** — sem colapsar o DataFrame (diferente do `groupBy`).

```mermaid
gantt
    title Janela deslizante de 7 dias — país Brasil
    dateFormat YYYY-MM-DD
    axisFormat %d/%m

    section Dia 10 (resultado = média dos 7)
    Dia 4  :done, d1, 2021-01-04, 1d
    Dia 5  :done, d2, 2021-01-05, 1d
    Dia 6  :done, d3, 2021-01-06, 1d
    Dia 7  :done, d4, 2021-01-07, 1d
    Dia 8  :done, d5, 2021-01-08, 1d
    Dia 9  :done, d6, 2021-01-09, 1d
    Dia 10 :active, d7, 2021-01-10, 1d
```

**`unix_date()` em vez de `.cast("long")`:** O Spark 4.x não permite converter `DateType` para `BIGINT`. A função `unix_date()` retorna dias desde 1970-01-01 como `IntegerType`, compatível com `Window.orderBy()`.

---

### Módulo 6 — Agregações Analíticas

```mermaid
graph LR
    DF["DataFrame\nLimpo e Enriquecido"] --> P["por País\n• total_cases\n• total_deaths\n• CFR\n• vacinação"]
    DF --> C["por Continente\n• soma de casos\n• mortes\n• vacinação média"]
    DF --> M["Mensal Global\n• soma de novos casos\n• soma de mortes\n• Rt médio"]
```

---

### Módulo 7 — Carga (Load)

```python
df.coalesce(1).write.mode("overwrite").option("header", "true").csv(destino)
```

- `coalesce(1)` — consolida todas as partições em um único arquivo CSV
- `mode("overwrite")` — substitui execuções anteriores automaticamente
- Saída em `data/processado/` (não versionado no Git por exceder 100 MB)

---

## 4. Notebook EDA — `notebook/main.ipynb`

```mermaid
flowchart TD
    S1["Seção 1\nConfiguração e Extração\nSparkSession + CSV"]
    S2["Seção 2\nVisão Geral\nSchema + estatísticas"]
    S3["Seção 3\nQualidade dos Dados\nGráfico de nulos"]
    S4["Seção 4\nTransformação\nWindow Function + cache"]
    S5["Seção 5\nAnálise Global\nSérie temporal"]
    S6["Seção 6\nTop Países\nCasos, mortes, por milhão"]
    S7["Seção 7\nAnálise por Continente"]
    S8["Seção 8\nVacinação Global"]
    S9["Seção 9\nFatores Socioeconômicos\nCFR, correlação, PIB"]
    S10["Seção 10\nIndicadores Epidemiológicos\nRt e índice de rigidez"]

    S1 --> S2 --> S3 --> S4
    S4 --> S5 --> S6 --> S7 --> S8 --> S9 --> S10
```

**Por que `df.cache()`?**
Após a transformação, o DataFrame é reutilizado em ~10 células. Sem cache, o Spark recalcularia toda a transformação a cada consulta. Com cache, armazena o resultado em memória após a primeira computação.

---

## 5. As 15 Visualizações

### Figura 01 — Proporção de Valores Nulos
**Tipo:** Barras horizontais | **Código:** `F.col(c).isNull().cast("int")`

Antes de qualquer análise é fundamental entender **o que não temos**. Esta figura documenta as limitações do dataset. `icu_patients` e `hosp_patients` têm 93% de nulos — dados hospitalares só foram reportados por países de alta renda.

---

### Figura 02 — Evolução Global Diária de Casos e Mortes
**Tipo:** Área + média móvel 7 dias, dois subgráficos

Visualização central da análise. Identifica as **ondas da pandemia** e relaciona com variantes (Delta, Ômicron) e vacinação. A área mostra a variação diária bruta; a linha é a média móvel (tendência real).

**Achado:** O pico de Janeiro/2022 (Ômicron) teve muito mais casos que os anteriores, mas mortalidade proporcionalmente menor — evidência do efeito vacinal.

---

### Figura 03 — Evolução Mensal Global
**Tipo:** Barras (casos) + eixo secundário linha (mortes)

O gráfico diário é muito ruidoso para padrões de longo prazo. A visão mensal revela claramente as fases da pandemia com contexto temporal.

---

### Figura 04 — Top 15 Países por Total de Casos
**Tipo:** Barras horizontais por continente

Ranking absoluto de países mais afetados. Cria contexto de escala antes das figuras normalizadas. Favorece países grandes — justificativa para a figura 06.

---

### Figura 05 — Top 15 Países por Total de Mortes
**Tipo:** Barras horizontais

Comparar os dois rankings (casos e mortes) revela diferenças na capacidade de resposta de cada sistema de saúde. O Brasil aparece no top 3 em mortes mesmo com menos casos que EUA — indicando CFR mais alta.

---

### Figura 06 — Top 15 Países — Mortes por Milhão
**Tipo:** Barras horizontais

Métrica normalizada por população para comparação justa entre países. Países pequenos da Europa (Peru, Bulgária) que não aparecem nos rankings absolutos surgem aqui como muito impactados.

```mermaid
graph LR
    A["País A\n1.000 mortes\n10M habitantes\n= 100 mortes/milhão"]
    B["País B\n10.000 mortes\n100M habitantes\n= 100 mortes/milhão"]
    C["Impacto IGUAL\npor habitante"]
    A --> C
    B --> C
```

---

### Figura 07 — Análise por Continente
**Tipo:** Dois gráficos de barras lado a lado (absoluto vs. normalizado)

A comparação dupla mostra como a escolha da métrica muda a narrativa. América do Norte lidera em absolutos; outros continentes se destacam quando normalizado.

---

### Figura 08 — Evolução Mensal de Casos por Continente
**Tipo:** Linhas múltiplas

As ondas não foram simultâneas — Europa foi afetada antes da América do Sul na primeira onda. A Ásia teve padrões diferentes devido à política zero-COVID da China.

---

### Figura 09 — Vacinação Global ao Longo do Tempo
**Tipo:** Área + média móvel

Cria o **antes e depois**: a queda das mortes em 2021-2022 coincide visualmente com o aumento da vacinação. O pico foi ~40 milhões de doses/dia globalmente em meados de 2021.

---

### Figura 10 — Top 20 Países por Cobertura Vacinal
**Tipo:** Barras horizontais com linha de meta (70% OMS)

A OMS estabeleceu 70% como meta para imunidade coletiva. A linha torna imediatamente visível quais países atingiram a meta. Países com < 1M habitantes foram excluídos para evitar distorções.

---

### Figura 11 — CFR por Continente (Boxplot)
**Tipo:** Boxplot

```
     caixa = 50% dos países (Q1 a Q3)
     linha central = mediana
     pontos fora = outliers
```

CFR depende de capacidade de testagem (mais testes = CFR menor), qualidade do sistema de saúde e estrutura etária. Por isso varia tanto entre países.

---

### Figura 12 — Heatmap de Correlação Socioeconômica
**Tipo:** Heatmap com valores de correlação de Pearson

| Par | Correlação | Interpretação |
|---|---|---|
| PIB per capita × % Vacinados | Positiva | Países ricos vacinaram mais |
| PIB per capita × CFR | Negativa | Países ricos têm menor mortalidade |
| Expectativa de Vida × Casos/Milhão | Positiva | Países mais velhos notificaram mais |

> Correlação ≠ Causalidade.

---

### Figura 13 — PIB per Capita vs. Cobertura Vacinal
**Tipo:** Scatter com escala logarítmica no eixo X

PIB varia de ~$500 a ~$100.000 — escala log distribui melhor os países. A tendência confirma que países mais ricos tiveram maior cobertura vacinal.

---

### Figura 14 — Taxa de Reprodução (Rt) Global
**Tipo:** Linha com intervalo interquartil

```
Rt = 1.5  →  pandemia crescendo (cada infectado transmite para 1,5 pessoas)
Rt = 1.0  →  pandemia estável
Rt = 0.8  →  pandemia recuando
```

A linha vermelha em Rt = 1 é o **limiar crítico**. A faixa mostra variação entre países — enquanto alguns controlavam, outros ainda cresciam no mesmo período.

---

### Figura 15 — Índice de Rigidez por Continente
**Tipo:** Linhas múltiplas ao longo do tempo

O Stringency Index (Oxford) é um índice de 0 a 100 que agrega: fechamento de escolas, comércio, eventos, restrições de viagem e obrigatoriedade de máscara.

- Pico em Abril/2020 — lockdowns iniciais
- Redução progressiva ao longo de 2021-2022 com a vacinação
- Ásia manteve restrições por mais tempo que outros continentes

---

## 6. Conceitos de Big Data Aplicados

### MapReduce

```mermaid
flowchart LR
    subgraph MAP["FASE MAP"]
        direction TB
        P1["Partição 1\nBrasil, 500 casos"]
        P2["Partição 2\nItália, 3000 casos"]
        P3["Partição 3\nBrasil, 700 casos"]
        M1["(Brasil, 500)"]
        M2["(Itália, 3000)"]
        M3["(Brasil, 700)"]
        P1 --> M1
        P2 --> M2
        P3 --> M3
    end

    subgraph SHUFFLE["SHUFFLE"]
        S1["Brasil: [500, 700, ...]"]
        S2["Itália: [3000, ...]"]
    end

    subgraph REDUCE["FASE REDUCE"]
        R1["Brasil: soma = 1.200"]
        R2["Itália: soma = 3.000"]
    end

    MAP --> SHUFFLE --> REDUCE
```

O `groupBy("country").agg(F.sum("new_cases"))` executa este fluxo internamente — sem que o programador gerencie partições.

---

### Lazy Evaluation

```mermaid
sequenceDiagram
    participant Dev as Desenvolvedor
    participant Spark as Spark (Catalyst)
    participant Exec as Execução

    Dev->>Spark: df.filter(...)
    Note over Spark: Apenas registra — não executa
    Dev->>Spark: df.groupBy(...).agg(...)
    Note over Spark: Apenas registra — não executa
    Dev->>Spark: df.show() / df.collect()
    Note over Spark: AGORA otimiza e executa tudo
    Spark->>Exec: Plano otimizado
    Exec-->>Dev: Resultado
```

O Spark não executa nada até uma **ação**. Isso permite ao Catalyst Optimizer reorganizar, combinar e eliminar operações redundantes.

---

### DataFrame API vs RDD

```mermaid
graph TD
    subgraph RDD["Nível Baixo — RDD (Spark 1.x)"]
        R["rdd.map(lambda r: (r['country'], r['new_cases']))\n.reduceByKey(lambda a,b: a+b)"]
    end
    subgraph DF["Nível Alto — DataFrame API (Spark 2.x+)"]
        D["df.groupBy('country').agg(F.sum('new_cases'))"]
    end
    DF -- "mais legível\nmesmo desempenho\n(Catalyst Optimizer)" --> RES["Mesmo resultado"]
    RDD --> RES
```

---

## 7. Guia de Apresentação em Sala

### Roteiro sugerido (15-20 minutos)

```mermaid
timeline
    title Roteiro de Apresentação — PySpark Real
    section 0-3 min
        Contextualização : O que é Big Data?
                         : Por que Spark?
                         : O dataset OWID
    section 3-7 min
        Pipeline ETL : Diagrama ETL
                     : Extract-Transform-Load
                     : Demonstrar main.py rodando
    section 7-14 min
        Visualizações : Fig 02 - ondas da pandemia
                      : Fig 06 - normalização por milhão
                      : Fig 11 - CFR por continente
                      : Fig 12 - correlação socioeconômica
                      : Fig 14 - Rt e limiar crítico
    section 14-20 min
        Conceitos : MapReduce com o diagrama
                  : Window Function com exemplo
                  : Perguntas
```

### Perguntas frequentes e respostas

| Pergunta | Resposta |
|---|---|
| "Por que Spark e não Pandas?" | Pandas carrega tudo em memória de uma máquina. Spark processa em partições distribuídas — o mesmo código escala para bilhões de linhas em cluster sem reescrever nada. |
| "Por que Window Function para média móvel?" | Demonstra SQL analítico que existe em todos os bancos modernos (PostgreSQL, BigQuery, Snowflake). É uma habilidade transferível para qualquer plataforma. |
| "O dataset é confiável?" | É do Our World in Data (Oxford/Global Change Data Lab), referência global. Limitação: países com pouca testagem têm dados subestimados. |
| "Por que o Spark rodou localmente?" | Spark funciona em modo local para desenvolvimento. Em produção, o mesmo código rodaria em AWS EMR, Databricks ou Google Dataproc sem modificações. |
| "CFR alto = sistema de saúde ruim?" | Não necessariamente — CFR alto pode indicar poucos testes (subnotificação). Precisa ser lido com contexto. |

---

*Documentação — EDA COVID-19 com PySpark Real — UVV — Abril/2026*
