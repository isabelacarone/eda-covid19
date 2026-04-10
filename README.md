# Projeto de Processamento de Grande Volumes de Dados - EDA COVID-19 com Apache Spark
## Universidade de Vila Velha

## eda-covid-19

EDA de dataset público, Our World in Data, com dados da COVID-19 utilizando Apache Spark para a disciplina de Processamento de Grande Volumes de Dados.

---

### Sobre o Java

O PySpark requer o **Java Development Kit (JDK) 17 ou superior** com o módulo `jdk.incubator.vector`.

O projeto inclui o pacote `install-jdk`, que **baixa e instala o JDK automaticamente** dentro do ambiente Python.

Para instalar o JDK antes de executar o projeto pela primeira vez:

```bash
# Com venv ativo
python -c "import jdk; jdk.install('21')"
```

O código detecta o JDK automaticamente a partir dessa localização.

---

### Executando utilizando venv

```bash
python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Instala o JDK (apenas na primeira vez)
python -c "import jdk; jdk.install('21')"
```

### Executando utilizando anaconda

```bash
conda create -n bigdata_env python=3.12
conda activate bigdata_env
pip install -r requirements.txt

# Instala o JDK via conda (já inclui jdk.incubator.vector)
conda install -c conda-forge openjdk=17
```

---

### Como executar o pipeline ETL

```bash
# Com venv ativo
python src/main.py
```

### Como executar o notebook

```bash
# Com venv ativo
jupyter notebook notebook/main.ipynb
# ou
jupyter lab
```

### Fazer download do dataset original

O dataset não está versionado no repositório (excede 100 MB). Faça o download antes de executar:

```bash
wget -O data/owid-covid.csv "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
```

---

### Estrutura

```
eda-covid-19/
├── notebook/
│   └── main.ipynb          # Análise exploratória 
├── src/
│   └── main.py             # Pipeline ETL completo com PySpark
├── data/
│   ├── owid-covid.csv      # Dataset original (não versionado por exceder 100 MB)
│   └── processado/         # Saídas do pipeline ETL
├── figuras/                # Gráficos gerados pelo notebook
├── .gitignore
├── venv/
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

### Referências

- <https://spark.apache.org/docs/4.0.1/api/python/getting_started/index.html>
- <https://pandas.pydata.org/docs/>
- <https://matplotlib.org/stable/contents.html>
- <https://seaborn.pydata.org/tutorial.html>
- <https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv>
- <https://pypi.org/project/install-jdk/>
