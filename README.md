# Projeto de Processamento de Grande Volumes de Dados - *EDA COVID-19*
## Universidade de Vila Velha

## eda-covid-19

EDA de dataset público, Our World in Data, com dados da COVID-19 utilizando Apache Spark e sua simulação para a disciplina de Processamento de Grande Volumes de Dados.

O projeto possui **duas implementações**:

| Implementação | Arquivos | Descrição |
|---|---|---|
| **PySpark** | `src/main.py` + `notebook/main.ipynb` | Pipeline ETL  |


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

### Como executar os notebooks

```bash
# Com venv ativo
jupyter notebook
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
│   ├── main.ipynb              # EDA com PySpark 
├── src/
│   ├── main.py                 # Pipeline ETL com PySpark
├── data/
│   ├── owid-covid.csv          # Dataset original (+100 MB)
│   ├── processado/             # Saídas do pipeline PySpark real
├── figuras/                    # Gráficos gerados pelos notebooks
│   ├── *.png                   # Figuras do PySpark 
├── documentacao.md             # Documentação de apoio, PySpark
├── .gitignore
├── venv/
├── slide/
├── requirements.txt
└── README.md
```

### Integrantes 
SI6N

- Gisela Medeiros
- Isabela Carone 
- Laiza Faqueri
- Mateus Sarmento 
- Sthefany Alves

### Referências

- <https://spark.apache.org/docs/4.0.1/api/python/getting_started/index.html>
- <https://pandas.pydata.org/docs/>
- <https://matplotlib.org/stable/contents.html>
- <https://seaborn.pydata.org/tutorial.html>
- <https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv>
- <https://pypi.org/project/install-jdk/>
- <https://docs.owid.io/projects/etl/>
