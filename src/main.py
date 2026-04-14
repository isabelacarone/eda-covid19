
import os
from typing import Final, List

# =============================================================================
#                  CONFIGURAÇÃO AUTOMÁTICA DO JAVA_HOME
# O PySpark requer Java 17+ com o módulo jdk.incubator.vector (JDK completo).
# Esta seção detecta o Java automaticamente a partir de três fontes,
# nesta ordem de prioridade:
#   1. JAVA_HOME já definido no ambiente (ex: conda, configuração do usuário)
#   2. JDK instalado via install-jdk em ~/.jdk/
#   3. java disponível no PATH do sistema
# =============================================================================

if not os.environ.get("JAVA_HOME"):
    try:
        # install-jdk baixa o JDK completo (inclui jdk.incubator.vector)
        # e o instala em ~/.jdk/; aqui apenas localizamos o mais recente

        import jdk as _jdk_mod
        jdk_base = os.path.expanduser("~/.jdk")
        if os.path.isdir(jdk_base):
            entradas = sorted(
                [d for d in os.listdir(jdk_base) if os.path.isdir(os.path.join(jdk_base, d))],
                reverse=True,
            )
            if entradas:
                os.environ["JAVA_HOME"] = os.path.join(jdk_base, entradas[0])
    except ImportError:
        pass  

# install-jdk não disponível, depende do Java. Se ocorrer erro lembre de instalar o Java no venv ou conda (acredito que vai dar tudo certo)

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window


# =============================================================================
#                         CONFIGURAÇÕES GLOBAIS
# =============================================================================

# caminho raiz do projeto (acima de src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMINHO_DADOS = os.path.join(ROOT_DIR, "data", "owid-covid.csv")
CAMINHO_SAIDA = os.path.join(ROOT_DIR, "data", "processado")

# países ou regiões que representam agregações (não são países individuais, mas sim agrupamentos de países ou regiões) 
# e que estão presentes no dataset como registros especiais

REGIOES_AGREGADAS : Final[List[str]] = [
    "World", "Asia", "Europe", "Africa", "North America",
    "South America", "Oceania", "European Union",
    "High income", "Low income", "Lower middle income",
    "Upper middle income", "High-income countries",
]


# =============================================================================
#                         CRIAÇÃO DA SPARK SESSION
# =============================================================================

def criar_spark_session(
        nome_app: str = "EDA_COVID19"
    ) -> SparkSession:
    
    """
    Cria e retorna uma SparkSession configurada para execução local

    Args:
        nome_app (str): Nome da aplicação Spark

    Returns:
        SparkSession: Sessão inicializada
    """

    spark = (
        SparkSession.builder
        .appName(nome_app)
        # usa todas as cores disponíveis localmente
        .master("local[*]")
        # "evita" logs grandes durante a execução
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    # Exibe apenas erros no log do Spark, evita poluição no terminal
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# =============================================================================
#                             EXTRAÇÃO
# =============================================================================

def extrair_dados(
        spark: SparkSession, 
        caminho: str
    ) -> DataFrame:

    """
    Lê o CSV do dataset COVID-19 com inferência de schema

    Args:
        spark (SparkSession): Sessão Spark ativa.
        caminho (str): Caminho para o arquivo CSV.

    Returns:
        DataFrame: Dataset >bruto< carregado
    """

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("dateFormat", "yyyy-MM-dd")
        .csv(caminho)
    )

    # 'date' como DataType
    df = df.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))
    return df


def exibir_visao_geral(df) -> None:
    """
    Imprime informações básicas sobre o DataFrame carregado.

    Args:
        df: DataFrame PySpark bruto.
    """
    total_linhas = df.count()
    total_colunas = len(df.columns)
    data_min = df.agg(F.min("date")).collect()[0][0]
    data_max = df.agg(F.max("date")).collect()[0][0]
    total_paises = df.select("country").distinct().count()
    total_continentes = df.filter(
        F.col("continent").isNotNull()
    ).select("continent").distinct().count()

    print("=" * 60)
    print("visão geral do dataset")
    print("=" * 60)
    print(f"  Linhas        : {total_linhas:,}")
    print(f"  Colunas       : {total_colunas}")
    print(f"  Países        : {total_paises}")
    print(f"  Continentes   : {total_continentes}")
    print(f"  Período       : {data_min} → {data_max}")
    print()
    print("Schema:")
    df.printSchema()

# =============================================================================
#                   ANÁLISE DE QUALIDADE (nulos, negativos)
# =============================================================================

def analisar_nulos(df) -> None:
    """
    Calcula e exibe a proporção de valores nulos nas colunas principais.

    Args:
        df: DataFrame PySpark.
    """
    colunas_interesse = [
        "total_cases", "new_cases", "total_deaths", "new_deaths",
        "total_vaccinations", "people_fully_vaccinated",
        "hosp_patients", "icu_patients", "stringency_index",
        "reproduction_rate", "continent", "population",
    ]

    print("=" * 60)
    print("análises dos valores nulos em todas as colunas")
    print("=" * 60)

    total = df.count()
    # realiza a expressão de contagem de nulos para todas as colunas de uma vez só
    exprs_nulos = [
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in colunas_interesse
    ]
    resultado = df.agg(*exprs_nulos).collect()[0].asDict()

    for coluna, qtd_nulos in sorted(resultado.items(), key=lambda x: -x[1]):
        pct = qtd_nulos / total * 100
        print(f"  {coluna:<45}: {qtd_nulos:>8,}  ({pct:5.1f}%)")
    print()

# =============================================================================
#                               TRANSFORMAÇÃO
# =============================================================================

def transformar_dados(df):
    """
    Aplica limpeza e enriquecimento ao dataset bruto:
      - Remove registros de regiões/agregações (não países)
      - Remove valores negativos inválidos
      - Preenche nulos numéricos com 0 onde apropriado
      - Cria colunas derivadas: year_month, mortality_rate, cases_per_100k

    Args:
        df: DataFrame PySpark bruto.

    Returns:
        DataFrame: Dataset limpo e enriquecido.
    """
    print("=" * 60)
    print("transformação dos dados")
    print("=" * 60)

    # Remove regiões agregadas e mantém apenas países individuais
    df_paises = df.filter(F.col("continent").isNotNull())
    print(f"  Registros após remover regiões: {df_paises.count():,}")

    # Remove registros com valores negativos (erro de notificação)
    df_limpo = (
        df_paises
        .filter(
            (F.col("new_cases").isNull()) | (F.col("new_cases") >= 0)
        )
        .filter(
            (F.col("new_deaths").isNull()) | (F.col("new_deaths") >= 0)
        )
    )
    print(f"  Registros após remover negativos: {df_limpo.count():,}")

    # Colunas derivadas
    df_enriquecido = (
        df_limpo
        # periodo mensaal
        .withColumn(
            "year_month",
            F.date_format(F.col("date"), "yyyy-MM")
        )
        # ano
        .withColumn("year", F.year(F.col("date")))
        # taxa de mortalidade diária (mortes / casos * 100)
        .withColumn(
            "daily_mortality_rate",
            F.when(
                (F.col("new_cases") > 0) & F.col("new_deaths").isNotNull(),
                (F.col("new_deaths") / F.col("new_cases") * 100).cast("double")
            ).otherwise(F.lit(None))
        )
        # casos por 100 mil habitantes
        .withColumn(
            "new_cases_per_100k",
            F.when(
                (F.col("population") > 0) & F.col("new_cases").isNotNull(),
                (F.col("new_cases") / F.col("population") * 100_000).cast("double")
            ).otherwise(F.lit(None))
        )
        # proporção de vacinados (dose completa / população)
        .withColumn(
            "fully_vaccinated_pct",
            F.when(
                (F.col("population") > 0) & F.col("people_fully_vaccinated").isNotNull(),
                (F.col("people_fully_vaccinated") / F.col("population") * 100).cast("double")
            ).otherwise(F.lit(None))
        )
    )

    print("Colunas adicionadas: year_month, year, daily_mortality_rate, new_cases_per_100k, fully_vaccinated_pct")
    print()
    return df_enriquecido

# =============================================================================
#                           'WINDOW FUNCTION'
# =============================================================================

def adicionar_media_movel(
        df, 
        coluna: str = "new_cases",
        janela: int = 7
     ) -> DataFrame:
    """
    Calcula a média móvel de uma coluna numérica particionada por país.

    Equivalente Spark:
        windowSpec = Window.partitionBy("country").orderBy("date")
                          .rowsBetween(-(janela-1), 0)
        df.withColumn("moving_avg", avg(coluna).over(windowSpec))

    Args:
        df: DataFrame PySpark com coluna 'country' e 'date'.
        coluna (str): Coluna numérica para calcular a média.
        janela (int): Tamanho da janela em dias.

    Returns:
        DataFrame: Com coluna adicional '{coluna}_ma{janela}'.
    """
    nome_saida = f"{coluna}_ma{janela}"

    window_spec = (
        Window
        .partitionBy("country")
        .orderBy(F.unix_date(F.col("date")))
        .rowsBetween(-(janela - 1), 0)
    )

    return df.withColumn(nome_saida, F.avg(F.col(coluna)).over(window_spec))



# =============================================================================
#                                   AGREGAÇÕES
# =============================================================================

def agregar_por_pais(df):
    """
    Sumariza métricas totais por país

    Args:
        df: DataFrame PySpark limpo.

    Returns:
        DataFrame: Uma linha por país com métricas agregadas.
    """
    return (
        df
        .groupBy("country", "continent", "population",
                 "gdp_per_capita", "median_age",
                 "life_expectancy", "human_development_index")
        .agg(
            F.max("total_cases").alias("total_cases"),
            F.max("total_deaths").alias("total_deaths"),
            F.max("people_fully_vaccinated").alias("people_fully_vaccinated"),
            F.avg("stringency_index").alias("avg_stringency_index"),
            F.avg("reproduction_rate").alias("avg_reproduction_rate"),
            F.count("date").alias("days_recorded"),
        )
        .withColumn(
            "case_fatality_rate",
            F.when(
                F.col("total_cases") > 0,
                (F.col("total_deaths") / F.col("total_cases") * 100)
            ).otherwise(F.lit(None))
        )
        .withColumn(
            "total_cases_per_million",
            F.when(
                F.col("population") > 0,
                (F.col("total_cases") / F.col("population") * 1_000_000)
            ).otherwise(F.lit(None))
        )
        .withColumn(
            "total_deaths_per_million",
            F.when(
                F.col("population") > 0,
                (F.col("total_deaths") / F.col("population") * 1_000_000)
            ).otherwise(F.lit(None))
        )
    )


def agregar_mensal_global(df):
    """
    Agrega novos casos e mortes por mês em nível global

    Args:
        df: DataFrame PySpark

    Returns:
        DataFrame: Uma linha por mês com totais globais
    """
    return (
        df
        .groupBy("year_month")
        .agg(
            F.sum("new_cases").alias("global_new_cases"),
            F.sum("new_deaths").alias("global_new_deaths"),
            F.avg("reproduction_rate").alias("avg_reproduction_rate"),
        )
        .orderBy("year_month")
    )


def agregar_por_continente(df):
    """
    Sumariza totais por continente.

    Args:
        df: DataFrame PySpark

    Returns:
        DataFrame: Uma linha por continente
    """
    return (
        df
        .groupBy("continent")
        .agg(
            F.max("total_cases").alias("total_cases"),
            F.max("total_deaths").alias("total_deaths"),
            F.avg("fully_vaccinated_pct").alias("avg_fully_vaccinated_pct"),
            F.avg("stringency_index").alias("avg_stringency_index"),
        )
        .orderBy(F.col("total_cases").desc())
    )


# =============================================================================
#                                  LOAD
# =============================================================================

def salvar_resultado(
        df,
        caminho: str, 
        nome: str, 
        formato: str = "csv"
    ) -> None:
    """
    Salva um DataFrame Spark em disco no formato especificado

    Args:
        df: DataFrame PySpark a salvar
        caminho (str): Diretório de saída
        nome (str): Subdiretório/nome do arquivo
        formato (str): Formato de saída 
    """
    destino = os.path.join(caminho, nome)
    (
        df.coalesce(1)   # Gera um único arquivo de saída
        .write
        .mode("overwrite")
        .option("header", "true")
        .format(formato)
        .save(destino)
    )
    print(f"  Salvo em: {destino}")

# =============================================================================
#                             rodada prévia
# =============================================================================
if __name__ == "__main__":
    spark = criar_spark_session()


    #                             EXTRACT

    print("\n" + "=" * 60)
    print("INICIANDO PIPELINE ETL: COVID-19")
    print("=" * 60 + "\n")

    df_bruto = extrair_dados(spark, CAMINHO_DADOS)
    exibir_visao_geral(df_bruto)
    analisar_nulos(df_bruto)


    #                           TRANSFORM
 
    df_limpo = transformar_dados(df_bruto)

    # Adiciona médias móveis de 7 dias para casos e mortes
    df_completo = adicionar_media_movel(df_limpo, "new_cases", 7)
    df_completo = adicionar_media_movel(df_completo, "new_deaths", 7)

    print("=" * 60)
    print("prévia dataset transformado")
    print("=" * 60)
    df_completo.select(
        "country", "date", "new_cases", "new_cases_ma7",
        "new_deaths", "new_deaths_ma7", "year_month"
    ).orderBy("country", "date").show(10, truncate=False)


    #                           AGREGAÇÕES
 
    print("=" * 60)
    print("top 10 países por total de casos")
    print("=" * 60)
    df_por_pais = agregar_por_pais(df_completo)
    df_por_pais.orderBy(F.col("total_cases").desc()).show(10, truncate=False)

    print("=" * 60)
    print("top 10 países por total de casos")
    print("=" * 60)
    df_por_pais = agregar_por_pais(df_completo)
    df_por_pais.orderBy(F.col("total_cases").desc()).show(10, truncate=False)

    print("=" * 60)
    print("totais por continente")
    print("=" * 60)
    agregar_por_continente(df_completo).show(truncate=False)

    print("=" * 60)
    print("evolução mensal global")
    print("=" * 60)
    agregar_mensal_global(df_completo).show(50, truncate=False)


    #                               LOAD
   
    print("=" * 60)
    print("salvando resultados processados...")
    print("=" * 60)
    os.makedirs(CAMINHO_SAIDA, exist_ok=True)
    salvar_resultado(df_por_pais, CAMINHO_SAIDA, "resumo_por_pais")
    salvar_resultado(agregar_mensal_global(df_completo), CAMINHO_SAIDA, "evolucao_mensal_global")
    salvar_resultado(agregar_por_continente(df_completo), CAMINHO_SAIDA, "resumo_por_continente")

    print("\n" + "=" * 60)
    print("pipeline concluída com sucesso \o/ ")
    print("=" * 60 + "\n")

    spark.stop()
