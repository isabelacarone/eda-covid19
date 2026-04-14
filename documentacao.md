# Documentação Técnica — EDA COVID-19 com Apache Spark (PySpark Real)

**Disciplina:** Processamento de Grande Volume de Dados — UVV
**Dataset:** Our World in Data — COVID-19 (~570 mil registros, 61 colunas)
**Período coberto:** Janeiro/2020 → Fevereiro/2026
**Implementação:** Apache Spark real via PySpark 4.1.1 + Java 21

> Esta documentação cobre exclusivamente os arquivos que utilizam o **Apache Spark de verdade**:
> `src/main.py` e `notebook/main.ipynb`.
> Para a simulação em Python puro, consulte `DOCUMENTACAO_SIMULACAO.md`.

---

## Visão Geral

O projeto realiza uma **Análise Exploratória de Dados (EDA)** sobre a pandemia de COVID-19 usando o **Apache Spark** (via PySpark) para processar um dataset de grande volume.

| Arquivo | Papel |
|---|---|
| `src/main.py` | Pipeline ETL |
| `notebook/main.ipynb` | Análise interativa com visualizações |

---

## 2. Arquitetura e Fluxo de Dados

```mermaid
flowchart 

    subgraph ENTRADA["input"]
        A[("owid-covid.csv")]
    end

    subgraph SPARK["APACHE SPARK, PySpark 4.1.1"]
        direction TB
        B["SparkSession<br>local[*]"]
        C["DataFrame Bruto (schema inferido)"]
        D["Filtragem: <br> remove regiões agregadas <br> remove valores negativos"]
        E["Enriquecimento: <br> year_month <br> daily_mortality_rate <br> fully_vaccinated_pct"]
        F["Window Function: <br> média móvel 7 dias <br> particionada por país"]
        G["Agregações: <br> por país, por continente e mensal global"]
    end

    subgraph SAIDA["output"]
        H[("data/processado/")]
        I["notebook/main.ipynb"]
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
    E["E , Extract<br>Lê o CSV bruto<br>com Spark"] --> T["T , Transform<br>Limpa, filtra<br>e enriquece"] --> L["L , Load<br>Salva CSVs<br>processados"]
```

---

### Configuração do Ambiente 

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

### Extração

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

### Análise de Qualidade dos Dados

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

### Transformação

```mermaid
flowchart TD
    A["DataFrame Bruto<br>570.606 linhas"] --> B

    B{"Tem continente<br>preenchido?"}
    B -- Não --> X1["Remove<br>(região agregada:<br>World, Asia, etc.)"]
    B -- Sim --> C

    C{"new_cases ou<br>new_deaths < 0?"}
    C -- Sim --> X2["Remove<br>(erro de notificação)"]
    C -- Não --> D

    D["Adiciona colunas derivadas"]
    D --> E["year_month<br> ex: '2021-03'"]
    D --> F2["daily_mortality_rate<br>= mortes/casos × 100"]
    D --> G2["fully_vaccinated_pct<br>= vacinados/população × 100"]
    D --> H2["new_cases_per_100k<br>= casos/população × 100.000"]
```

O dataset OWID inclui registros como `"World"`, `"High income"`, `"European Union"` que são **somas de países** — mantê-los duplicaria os dados nas análises.

---

### Window Function 

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
    title Janela deslizante de 7 dias , país Brasil
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

### Agregações Analíticas

```mermaid
graph LR
    DF["DataFrame<br>Limpo e Enriquecido"] --> P["por País<br>• total_cases<br>• total_deaths<br>• CFR<br>• vacinação"]
    DF --> C["por Continente<br>• soma de casos<br>• mortes<br>• vacinação média"]
    DF --> M["Mensal Global<br>• soma de novos casos<br>• soma de mortes<br>• Rt médio"]
```

---

### Carga 

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
    S1["Seção 1<br>Configuração e Extração<br>SparkSession + CSV"]
    S2["Seção 2<br>Visão Geral<br>Schema + estatísticas"]
    S3["Seção 3<br>Qualidade dos Dados<br>Gráfico de nulos"]
    S4["Seção 4<br>Transformação<br>Window Function + cache"]
    S5["Seção 5<br>Análise Global<br>Série temporal"]
    S6["Seção 6<br>Top Países<br>Casos, mortes, por milhão"]
    S7["Seção 7<br>Análise por Continente"]
    S8["Seção 8<br>Vacinação Global"]
    S9["Seção 9<br>Fatores Socioeconômicos<br>CFR, correlação, PIB"]
    S10["Seção 10<br>Indicadores Epidemiológicos<br>Rt e índice de rigidez"]
    S11["Seção 11<br>Análises Epidemiológicas<br>Curva epidêmica, incidência, CFR temporal, mortalidade/100k, Stringency vs Rt"]

    S1 --> S2 --> S3 --> S4
    S4 --> S5 --> S6 --> S7 --> S8 --> S9 --> S10 --> S11
```

**Por que `df.cache()`?**
Após a transformação, o DataFrame é reutilizado em ~10 células. Sem cache, o Spark recalcularia toda a transformação a cada consulta. Com cache, armazena o resultado em memória após a primeira computação.

---


