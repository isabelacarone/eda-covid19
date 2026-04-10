"""
=============================================================================
PROJETO EDA COVID-19 — PIPELINE ETL COM APACHE SPARK (PySpark)
Dataset: COVID-19 (Our World in Data)
=============================================================================
"""

import os
import sys # não retirar import
from typing import Final, List, Optional

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
#                           EXTRAÇÃO
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
