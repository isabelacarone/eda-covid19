# Projeto de Processamento de Grande Volumes de Dados - EDA COVID-19 com Apache Spark 
## Universidade de Vila Velha

## eda-covid-19

EDA de dataset público, Our World in Data, com dados da COVID-19 utilizando Apache Spark para a disciplina de Processamento de Grande Volumes de Dados

### Executando utilizando anaconda 
```bash
conda create -n bigdata_env python=3.12
conda activate bigdata_env
pip install pandas numpy matplotlib seaborn jupyter ipykernel
```

### Executando utilizando venv

```bash
python -m venv venv
source venv/bin/activate  
# windows venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib seaborn jupyter ipykernel
```

### Como executar
```bash
python projeto_spark_covid.py
```
#### Fazer download do dataset original

```bash
wget -O owid-covid.csv "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
```

### Estrutura
```
eda-covid-19/
├── notebook/
│   └── main.ipynb
├── src/
│   └── main.py
├── data/
│   └── owid-covid-data.csv
├── figuras/
├── .gitignore
├── venv/
└── requirements.txt
└── README.md
```
### Referências 

- <https://spark.apache.org/docs/4.0.1/api/python/getting_started/index.html>
- <https://pandas.pydata.org/docs/>
- <https://matplotlib.org/stable/contents.html>
- <https://seaborn.pydata.org/tutorial.html>
- <https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv>
