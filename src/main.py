"""
=============================================================================
PROJETO EDA COVID-19 — PIPELINE ETL COM APACHE SPARK (PySpark)
Dataset: COVID-19 (Our World in Data)
=============================================================================

ETAPAS:
  1. Extração   --> leitura do CSV com SparkSession
  2. Exploração --> schema, contagens, valores nulos
  3. Transforma --> limpeza e criação de colunas derivadas
  4. Agregação  --> groupBy, window functions, análises globais
  5. Carga      --> exporta resultados processados em CSV
"""

import os
import sys # não retirar import

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window


# =============================================================================
#                           CONFIGURAÇÕES GLOBAIS
# =============================================================================

# Caminho raiz do projeto (um nível acima de src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMINHO_DADOS = os.path.join(ROOT_DIR, "data", "owid-covid.csv")
CAMINHO_SAIDA = os.path.join(ROOT_DIR, "data", "processado")

# Países/regiões que representam agregações (não países individuais),
# presentes no dataset como registros especiais

REGIOES_AGREGADAS = [
    "World", "Asia", "Europe", "Africa", "North America",
    "South America", "Oceania", "European Union",
    "High income", "Low income", "Lower middle income",
    "Upper middle income", "High-income countries",
]
