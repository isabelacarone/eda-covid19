# Apresentação — EDA COVID-19 com Apache Spark

> **Roteiro para o time** — cada seção indica o que falar, por que importa e qual trecho do código mostrar.

---

## 1. Contexto do Projeto

**O que falar:**
> "Desenvolvemos uma análise exploratória da pandemia de COVID-19 usando Apache Spark para processamento distribuído. O dataset vem do Our World in Data e cobre mais de 570 mil registros de 262 países entre janeiro de 2020 e fevereiro de 2026."

**Destaques:**
- Instituição: Universidade de Vila Velha — turma SI6N
- Time: Gisela Medeiros, Isabela Carone, Laiza Faqueri, Mateus Sarmento, Sthefany Alves
- Dataset: `owid-covid.csv` — 570.606 linhas × 61 colunas
- Duas entregas: pipeline ETL (`src/main.py`) + notebook interativo (`notebook/main.ipynb`)

---

## 2. Arquitetura e Fluxo de Dados

**O que falar:**
> "O projeto segue o padrão ETL clássico: extraímos o CSV, transformamos com Spark e carregamos três datasets processados. O notebook complementa com visualizações."

```
owid-covid.csv
      ↓
  [EXTRACT]  → leitura com Spark, inferência de schema
      ↓
 [TRANSFORM] → limpeza, filtragem, colunas derivadas, média móvel
      ↓
   [LOAD]    → data/processado/ (3 arquivos CSV)
      +
  notebook   → 20 visualizações em figuras/
```

**Mostrar no `documentacao.md`:** diagrama Mermaid da arquitetura (seção "Arquitetura e Fluxo de Dados").

---

## 3. Configuração do Ambiente

**O que falar:**
> "Usamos PySpark 4.1.1 com Java 17+. O código detecta automaticamente o JAVA_HOME para não precisar de configuração manual."

**Trecho relevante — `src/main.py` (início do arquivo):**
```python
# Detecta JAVA_HOME automaticamente: ~/.jdk/, conda ou PATH do sistema
for candidate in [Path.home() / ".jdk", ...]:
    if candidate.exists():
        os.environ["JAVA_HOME"] = str(candidate)
        break
```

**Configurações da SparkSession:**
```python
def criar_spark_session(nome_app="EDA_COVID19"):
    """Cria e retorna uma SparkSession configurada para processamento local."""
    return (
        SparkSession.builder
        .appName(nome_app)
        .master("local[*]")                          # usa todos os cores do CPU
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
```

> "O `local[*]` usa todos os núcleos do processador. O shuffle em 8 partições e 2 GB de memória são ajustados para o volume de dados."

---

## 4. EXTRACT — Leitura dos Dados

**O que falar:**
> "A extração lê o CSV com inferência de schema automática do Spark e converte a coluna de data para o tipo correto."

**Trecho — `src/main.py`, função `extrair_dados`:**
```python
def extrair_dados(spark, caminho):
    """
    Extrai dados do arquivo CSV para um DataFrame Spark.
    Args:
        spark: SparkSession ativa
        caminho: caminho para o arquivo CSV
    Returns:
        DataFrame com os dados brutos
    """
    df = spark.read.csv(caminho, header=True, inferSchema=True)
    df = df.withColumn("date", col("date").cast(DateType()))
    return df
```

**Resultado:** DataFrame com 570.606 linhas e 61 colunas.

---

## 5. Qualidade dos Dados — Análise de Nulos

**O que falar:**
> "Antes de transformar, analisamos a qualidade. Fizemos isso em uma única passagem no DataFrame, aproveitando a avaliação lazy do Spark."

**Trecho — `src/main.py`, função `analisar_nulos`:**
```python
def analisar_nulos(df):
    """
    Analisa percentual de valores nulos nas colunas principais.
    Usa agregação em passagem única para otimização com lazy evaluation.
    """
    total = df.count()
    exprs = [
        (F.sum(F.col(c).isNull().cast("int")) / total * 100).alias(c)
        for c in COLUNAS_PRINCIPAIS
    ]
    return df.agg(*exprs)
```

**Principais achados (mostrar figura `01_valores_nulos.png`):**

| Coluna | Nulos |
|---|---|
| `icu_patients` | ~93% |
| `hosp_patients` | ~93% |
| `people_fully_vaccinated` | ~87% |
| `reproduction_rate` | ~68% |
| `new_cases` | ~3% |

> "Colunas com alta taxa de nulos não são descartadas — são usadas onde têm dados, como nas análises de vacinação e taxa de reprodução."

---

## 6. TRANSFORM — Limpeza e Enriquecimento

**O que falar:**
> "A transformação tem duas etapas: limpeza (remover regiões agregadas e valores negativos) e enriquecimento (novas colunas derivadas)."

**Trecho — `src/main.py`, função `transformar_dados`:**
```python
def transformar_dados(df):
    """
    Aplica transformações de limpeza e enriquecimento ao DataFrame.
    Remove regiões agregadas (World, Asia, etc.) e valores negativos.
    Adiciona colunas derivadas: year_month, daily_mortality_rate,
    new_cases_per_100k, fully_vaccinated_pct.
    """
    df = df.filter(col("continent").isNotNull())           # remove agregações
    df = df.filter(col("new_cases") >= 0)                  # remove negativos
    df = df.withColumn("year_month", date_format(col("date"), "yyyy-MM"))
    df = df.withColumn(
        "daily_mortality_rate",
        when(col("new_cases") > 0,
             col("new_deaths") / col("new_cases") * 100).otherwise(None)
    )
    df = df.withColumn(
        "new_cases_per_100k",
        col("new_cases") / col("population") * 100_000
    )
    df = df.withColumn(
        "fully_vaccinated_pct",
        col("people_fully_vaccinated") / col("population") * 100
    )
    return df
```

**Resultado:** 528.390 registros limpos (−42.216 linhas removidas).

---

## 7. Média Móvel com Window Function

**O que falar:**
> "Para suavizar ruído nos dados diários — como subnotificações nos fins de semana — calculamos a média móvel de 7 dias usando Window Function do Spark."

**Trecho — `src/main.py`, função `adicionar_media_movel`:**
```python
def adicionar_media_movel(df, coluna="new_cases", janela=7):
    """
    Adiciona coluna de média móvel usando Window Function.
    Particiona por país e ordena por data para calcular
    a média dos últimos N dias (padrão: 7).
    Args:
        coluna: coluna base (ex: 'new_cases', 'new_deaths')
        janela: tamanho da janela em dias
    Returns:
        DataFrame com coluna '{coluna}_ma{janela}'
    """
    w = Window.partitionBy("location").orderBy("date") \
              .rowsBetween(-(janela - 1), 0)
    return df.withColumn(f"{coluna}_ma{janela}", avg(col(coluna)).over(w))
```

> "O `partitionBy('location')` garante que a janela não cruza fronteiras entre países."

**Mostrar figura `02_evolucao_global.png`** — efeito da suavização visível na curva.

---

## 8. Agregações — LOAD

**O que falar:**
> "Geramos três visões agregadas e salvamos como CSV. O `coalesce(1)` consolida as partições em um único arquivo de saída."

**Três agregações produzidas:**

**Por país — `agregar_por_pais`:**
```python
df.groupBy("location", "continent", "population").agg(
    sum("new_cases").alias("total_cases"),
    sum("new_deaths").alias("total_deaths"),
    (sum("new_deaths") / sum("new_cases") * 100).alias("case_fatality_rate"),
    (sum("new_cases") / first("population") * 1_000_000).alias("cases_per_million"),
)
```

**Mensal global — `agregar_mensal_global`:**
```python
df.groupBy("year_month").agg(
    sum("new_cases").alias("global_cases"),
    sum("new_deaths").alias("global_deaths"),
    sum("new_vaccinations").alias("global_vaccinations"),
)
```

**Por continente — `agregar_por_continente`:**
```python
df.groupBy("continent").agg(
    sum("new_cases").alias("total_cases"),
    avg("reproduction_rate").alias("avg_reproduction_rate"),
    avg("stringency_index").alias("avg_stringency_index"),
)
```

**Salvamento:**
```python
def salvar_resultado(df, caminho, nome, formato="csv"):
    """Salva DataFrame processado em disco. Coalesce para 1 partição."""
    df.coalesce(1).write.mode("overwrite") \
      .option("header", True).csv(f"{caminho}/{nome}")
```

---

## 9. Visualizações — Notebook

**O que falar:**
> "O notebook reproduz as mesmas transformações e gera 20 visualizações. Usamos uma paleta padronizada para consistência visual."

**Paleta de cores (notebook, célula de configuração):**
```python
COR_CASOS   = "#2196F3"   # azul
COR_MORTES  = "#F44336"   # vermelho
COR_VACINAS = "#4CAF50"   # verde
```

**Blocos de análise e figuras correspondentes:**

| Bloco | Figuras | Destaque |
|---|---|---|
| Dados globais | `02`, `03` | curva de casos + mortes com MM7 |
| Top países | `04`, `05`, `06` | ranking por casos, mortes, mortes/milhão |
| Continentes | `07`, `08` | total e evolução temporal |
| Vacinação | `09`, `10` | global + top 20 (meta OMS 70%) |
| Mortalidade | `11`, `12`, `13` | CFR, correlação socioeconômica, PIB vs vacinas |
| Epidemiologia | `14`, `15`, `16` | Rt, índice de rigidez, curva epidêmica com ondas |
| Incidência | `17`, `18`, `19`, `20` | por continente, CFR temporal, mortalidade/100k, rigidez vs Rt |

---

## 10. Análises de Destaque

### 10.1 Curva Epidêmica com Ondas (`16_curva_epidemica.png`)
> "Identificamos cinco fases da pandemia com base nos picos de casos semanais."

Ondas anotadas no gráfico:
- 1ª Onda (2020)
- 2ª Onda (2020–2021)
- 3ª Onda — Alpha/Delta (2021)
- 4ª Onda — Ômicron (2021–2022)
- Transição para endemia

### 10.2 Taxa de Reprodução Rt (`14_taxa_reproducao.png`)
> "O Rt abaixo de 1,0 indica controle epidêmico. Mostramos a média global com intervalo interquartil (Q25–Q75) sombreado."

### 10.3 Correlação Socioeconômica (`12_correlacao_socioeconomica.png`)
> "Exploramos como PIB per capita, idade mediana e expectativa de vida se correlacionam com impacto da COVID — países mais ricos tiveram maior vacinação mas também maior subnotificação inicial."

### 10.4 Rigidez de Políticas vs Rt (`20_stringency_vs_rt.png`)
> "Medimos a efetividade das políticas de contenção: correlação entre Stringency Index e taxa de reprodução por continente."

---

## 11. Tecnologias Utilizadas

| Tecnologia | Versão | Papel |
|---|---|---|
| Apache Spark / PySpark | 4.1.1 | processamento distribuído |
| Python | 3.13 | linguagem principal |
| pandas | 3.0.2 | manipulação em notebook |
| matplotlib + seaborn | 3.10 / 0.13 | visualizações |
| Jupyter / JupyterLab | 1.1 / 4.5 | notebook interativo |
| Java JDK | 17+ | runtime do Spark |

---




