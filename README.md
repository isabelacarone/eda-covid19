# Projeto de Processamento de Grande Volumes de Dados - *EDA COVID-19*
## Universidade de Vila Velha

## eda-covid-19

EDA de dataset público, Our World in Data, com dados da COVID-19 utilizando Apache Spark e sua simulação para a disciplina de Processamento de Grande Volumes de Dados.

O projeto possui **duas implementações paralelas**:

| Implementação | Arquivos | Descrição |
|---|---|---|
| **PySpark Real** | `src/main.py` + `notebook/main.ipynb` | Pipeline ETL e EDA com Apache Spark 4.1.1 + Java 21 |


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

> Já a  **simulação** (`simulacao.py` e `simulacao.ipynb`) **não precisa de Java**. Roda com Python puro.

---

### Executando utilizando venv

```bash
python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\Activate.ps1

pip install -r requirements.txt

# para PySpark real
python -c "import jdk; jdk.install('21')"
```

### Executando utilizando anaconda

```bash
conda create -n bigdata_env python=3.12
conda activate bigdata_env
pip install -r requirements.txt

conda install -c conda-forge openjdk=17
```

---

### Como executar o pipeline ETL com PySpark

```bash
# Com venv ativo
python src/main.py
```

Saídas geradas em `data/processado/`.

### Como executar a simulação do Spark

```bash
# Com venv ativo (não precisa de Java)
python src/simulacao.py
```

Saídas geradas em `data/processado_sim/`.

### Como executar os notebooks

```bash
# Com venv ativo
jupyter notebook
# ou
jupyter lab
```

- `notebook/main.ipynb` =>  EDA com PySpark
- `notebook/simulacao.ipynb` => EDA com a simulação do Spark 

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
│   ├── main.ipynb              # EDA com PySpark 
│   └── simulacao.ipynb         # EDA com simulação Python
├── src/
│   ├── main.py                 # Pipeline ETL com PySpark rea?
│   └── main_simulado.py        # Simulação do Spark em Python 
├── data/
│   ├── owid-covid.csv          # Dataset original (+100 MB)
│   ├── processado/             # Saídas do pipeline PySpark real
│   └── processado_sim/         # Saídas da simulação
├── figuras/                    # Gráficos gerados pelos notebooks
│   ├── *.png                   # Figuras do PySpark 
│   └── *_sim.png               # Figuras da simulação
├── documentacao.md             # Documentação de apoio, PySpark
├── documentacao_simulacao.md   # Documentação de apoio, simulação 
├── .gitignore
├── venv/
├── requirements.txt
└── README.md
```

### Referências

- <https://spark.apache.org/docs/4.0.1/api/python/getting_started/index.html>
- <https://pandas.pydata.org/docs/>
- <https://matplotlib.org/stable/contents.html>
- <https://seaborn.pydata.org/tutorial.html>
- <https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv>
- <https://pypi.org/project/install-jdk/>
- <https://docs.owid.io/projects/etl/>
