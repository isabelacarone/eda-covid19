"""
=============================================================================
SIMULAÇÃO DO APACHE SPARK EM PYTHON PURO — EDA COVID-19
Dataset: COVID-19 (Our World in Data)
Curso: Processamento de Grande Volume de Dados — UVV
=============================================================================

Este arquivo implementa uma simulação simplificada dos principais conceitos
do Apache Spark, usando apenas Python, Pandas e NumPy — sem nenhuma
dependência do PySpark.

CONCEITOS SIMULADOS:
  1. RDD             — Resilient Distributed Dataset com partições em memória
  2. Map / Filter    — transformações elemento a elemento sobre partições
  3. ReduceByKey     — agregação por chave (coração do MapReduce)
  4. DataFrame API   — groupBy, agg, filter, withColumn, select, orderBy
  5. Column          — expressões de coluna tipadas (operadores, null-checks)
  6. Window Function — média móvel por partição ordenada por data
  7. SparkSession    — ponto de entrada único que orquestra tudo
  8. Lazy Evaluation — transformações registradas; execução adiada para ação

ESTRUTURA DO ARQUIVO:
  Parte 1 (linhas ~40-380)  — Engine de simulação (classes reutilizáveis)
  Parte 2 (linhas ~380-fim) — Pipeline ETL aplicado ao dataset COVID-19
"""

import os
import math
from collections import defaultdict

import numpy as np
import pandas as pd


# =============================================================================
#                            simulação do apache spark!
# =============================================================================


#                             caminhos do projeto


ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_CSV = os.path.join(ROOT_DIR, "data", "owid-covid.csv")
CAMINHO_SAIDA = os.path.join(ROOT_DIR, "data", "processado_sim")

# protect: garante que qualquer valor vire uma Column 

def _to_col(x):
    """
    Converte uma string (nome de coluna) ou valor literal em um objeto Column.
    Permite escrever F.sum("new_cases") ou F.sum(F.col("new_cases")) com o
    mesmo resultado — como no PySpark real.
    """
    if isinstance(x, (Column, WhenBuilder)):
        return x
    if isinstance(x, str):
        # Captura 'x' por valor com argumento default para evitar closure tardio
        return Column(lambda df, _n=x: df[_n], x)
    val = x
    return Column(lambda df, _v=val: pd.Series([_v] * len(df), index=df.index), str(val))


# =============================================================================
# SEÇÃO 1.1 — COLUMN: expressões de coluna
# =============================================================================

class Column:
    """
    Representa uma expressão de coluna — análogo ao pyspark.sql.Column.

    Internamente armazena uma função (eval_fn) que recebe um DataFrame
    pandas e retorna uma pandas Series com o resultado da expressão.
    Esse padrão simula a avaliação preguiçosa (lazy evaluation) do Spark:
    a expressão só é calculada quando realmente necessário.
    """

    def __init__(self, eval_fn, name="expr"):
        self._eval_fn = eval_fn
        self._name    = name

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Avalia a expressão sobre um DataFrame pandas e retorna uma Series."""
        return self._eval_fn(df)

    def alias(self, name: str) -> "Column":
        """Renomeia o resultado — equivalente ao .alias() do PySpark."""
        return Column(self._eval_fn, name)

    def cast(self, dtype: str) -> "Column":
        """Converte o tipo da coluna — equivalente ao .cast() do PySpark."""
        def _cast(df, _self=self):
            s = _self.evaluate(df)
            if dtype in ("double", "float"):
                return pd.to_numeric(s, errors="coerce")
            if dtype in ("int", "integer", "bigint", "long"):
                return pd.to_numeric(s, errors="coerce")
            if dtype == "string":
                return s.astype(str)
            return s
        return Column(_cast, f"CAST({self._name} AS {dtype})")

    # os valores podems ser...
    def isNull(self) -> "Column":
        return Column(lambda df, _s=self: _s.evaluate(df).isna(),
                      f"isnull({self._name})")

    def isNotNull(self) -> "Column":
        return Column(lambda df, _s=self: ~_s.evaluate(df).isna(),
                      f"isnotnull({self._name})")
    # fim

    def isin(self, *values) -> "Column":
        vals = list(values)
        return Column(lambda df, _s=self, _v=vals: _s.evaluate(df).isin(_v),
                      f"{self._name} IN ({vals})")

    # operadores aritméticos
    def __add__(self, other):
        o = _to_col(other)
        return Column(lambda df, _a=self, _b=o: _a.evaluate(df) + _b.evaluate(df),
                      f"({self._name}+{o._name})")

    def __radd__(self, other):
        return _to_col(other).__add__(self)

    def __sub__(self, other):
        o = _to_col(other)
        return Column(lambda df, _a=self, _b=o: _a.evaluate(df) - _b.evaluate(df),
                      f"({self._name}-{o._name})")

    def __mul__(self, other):
        o = _to_col(other)
        return Column(lambda df, _a=self, _b=o: _a.evaluate(df) * _b.evaluate(df),
                      f"({self._name}*{o._name})")

    def __rmul__(self, other):
        return _to_col(other).__mul__(self)

    def __truediv__(self, other):
        o = _to_col(other)
        return Column(lambda df, _a=self, _b=o: _a.evaluate(df) / _b.evaluate(df),
                      f"({self._name}/{o._name})")
    # fim

    # operadores de comparação
    def __gt__(self, other):
        o = _to_col(other)
        return Column(lambda df, _a=self, _b=o: _a.evaluate(df) > _b.evaluate(df),
                      f"({self._name}>{o._name})")

    def __ge__(self, other):
        o = _to_col(other)
        return Column(lambda df, _a=self, _b=o: _a.evaluate(df) >= _b.evaluate(df),
                      f"({self._name}>={o._name})")

    def __lt__(self, other):
        o = _to_col(other)
        return Column(lambda df, _a=self, _b=o: _a.evaluate(df) < _b.evaluate(df),
                      f"({self._name}<{o._name})")

    def __le__(self, other):
        o = _to_col(other)
        return Column(lambda df, _a=self, _b=o: _a.evaluate(df) <= _b.evaluate(df),
                      f"({self._name}<={o._name})")

    def __eq__(self, other):
        o = _to_col(other)
        return Column(lambda df, _a=self, _b=o: _a.evaluate(df) == _b.evaluate(df),
                      f"({self._name}=={o._name})")

    def __ne__(self, other):
        o = _to_col(other)
        return Column(lambda df, _a=self, _b=o: _a.evaluate(df) != _b.evaluate(df),
                      f"({self._name}!={o._name})")
    # fim

    # operadores booleanos
    def __and__(self, other):
        return Column(lambda df, _a=self, _b=other: _a.evaluate(df) & _b.evaluate(df),
                      f"({self._name} AND {other._name})")

    def __or__(self, other):
        return Column(lambda df, _a=self, _b=other: _a.evaluate(df) | _b.evaluate(df),
                      f"({self._name} OR {other._name})")

    def __invert__(self):
        return Column(lambda df, _s=self: ~_s.evaluate(df), f"NOT({self._name})")

    def __repr__(self):
        return f"Column('{self._name}')"
    # fim


#                           expressões de agregação


class AggExpr:
    """
    Expressão de agregação (sum, avg, max, min, count, etc.) — análoga a
    F.sum("col").alias("total") no PySpark.

    Usada dentro de GroupedData.agg() para calcular a agregação
    correspondente sobre um pandas GroupBy.
    """

    def __init__(self, func_name: str, col, alias_name: str = None, **kwargs):
        self.func_name  = func_name
        self.col        = col          # Column, string, ou None (para count*)
        self._alias     = alias_name
        self.kwargs     = kwargs       # ex: percentile={'p': 0.5}

    def alias(self, name: str) -> "AggExpr":
        return AggExpr(self.func_name, self.col, alias_name=name, **self.kwargs)

    def __repr__(self):
        nome = self._alias or f"{self.func_name}({self.col})"
        return f"AggExpr({nome})"


# 
#                       expressões condicionais (when/otherwise)
# 

class WhenBuilder:
    """
    Constrói uma expressão condicional encadeada — análoga ao
    F.when(cond, val).when(cond2, val2).otherwise(default) do PySpark.
    """

    def __init__(self, condition, value):
        self._branches  = [(condition, _to_col(value))]
        self._otherwise = None
        self._name      = "when(...)"

    def when(self, condition, value) -> "WhenBuilder":
        self._branches.append((condition, _to_col(value)))
        return self

    def otherwise(self, value) -> "WhenBuilder":
        self._otherwise = _to_col(value)
        return self

    def alias(self, name: str) -> "Column":
        """Converte o WhenBuilder em um Column com nome definido."""
        return Column(self.evaluate, name)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Avalia a expressão condicional sobre um DataFrame pandas."""
        result = pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
        # Percorre as branches em ordem; a primeira que satisfaz vence
        for condition, value_col in self._branches:
            cond_series = (condition.evaluate(df)
                           if isinstance(condition, Column) else condition)
            val_series  = value_col.evaluate(df)
            # Preenche apenas onde ainda é NaN e a condição é True
            mask = cond_series & result.isna()
            result = result.where(~mask, val_series)
        if self._otherwise is not None:
            null_mask = result.isna()
            result = result.where(~null_mask, self._otherwise.evaluate(df))
        return result

    def __repr__(self):
        return f"WhenBuilder({len(self._branches)} branches)"


# =============================================================================
# SEÇÃO 1.4 — WINDOW FUNCTION
# =============================================================================

class WindowSpec:
    """
    Especificação de janela para funções analíticas — análoga ao
    Window.partitionBy(...).orderBy(...).rowsBetween(...) do PySpark.
    """

    def __init__(self, partition_cols=None, order_cols=None,
                 rows_start=None, rows_end=None):
        self.partition_cols = partition_cols or []
        # Armazena os nomes E as expressões de ordenação separadamente.
        # As expressões são necessárias quando orderBy recebe um Column
        # (ex: F.unix_date(F.col("date"))) que precisa ser avaliado antes do sort.
        order_items = order_cols or []
        self.order_cols      = [c if isinstance(c, str) else c._name
                                 for c in order_items]
        self.order_col_exprs = [None if isinstance(c, str) else c
                                 for c in order_items]
        self.rows_start      = rows_start
        self.rows_end        = rows_end

    def orderBy(self, *cols) -> "WindowSpec":
        return WindowSpec(self.partition_cols, list(cols),
                          self.rows_start, self.rows_end)

    def rowsBetween(self, start: int, end: int) -> "WindowSpec":
        # Reconstrói preservando as expressões de ordenação originais
        raw = [expr if expr is not None else name
               for name, expr in zip(self.order_cols, self.order_col_exprs)]
        return WindowSpec(self.partition_cols, raw, start, end)


class Window:
    """
    Ponto de entrada para criar especificações de janela —
    análogo ao pyspark.sql.Window.
    """

    @staticmethod
    def partitionBy(*cols) -> WindowSpec:
        """Cria uma WindowSpec particionada pelas colunas especificadas."""
        return WindowSpec(partition_cols=list(cols))


class WindowColumn:
    """
    Coluna com função de janela — resultado de agg_expr.over(window_spec).
    Armazena a expressão de agregação e a especificação de janela;
    o cálculo real é feito dentro de SimulatedDataFrame.withColumn().
    """

    def __init__(self, agg_expr: AggExpr, window_spec: WindowSpec):
        self.agg_expr    = agg_expr
        self.window_spec = window_spec
        self._name       = f"{agg_expr.func_name}_over_window"

    def alias(self, name: str) -> "WindowColumn":
        wc = WindowColumn(self.agg_expr, self.window_spec)
        wc._name = name
        return wc

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula a função de janela usando pandas groupby + rolling.
        Simula o comportamento do Window.partitionBy().orderBy().rowsBetween()
        do Spark: para cada grupo (partição), ordena por data e aplica a
        janela deslizante.
        """
        agg   = self.agg_expr
        spec  = self.window_spec
        result = pd.Series(index=df.index, dtype="float64")

        # Resolve a coluna alvo
        col = agg.col
        if isinstance(col, str):
            col_name = col
        elif isinstance(col, Column):
            col_name = f"__win_tmp_{agg.func_name}__"
            df = df.copy()
            df[col_name] = col.evaluate(df)
        else:
            raise ValueError(f"Tipo de coluna não suportado: {type(col)}")

        # Tamanho da janela deslizante
        if spec.rows_start is not None and spec.rows_end is not None:
            window_size = spec.rows_end - spec.rows_start + 1
        else:
            window_size = None

        # Adiciona colunas de ordenação que são expressões (ex: unix_date(date))
        # ao DataFrame antes de iterar pelas partições
        for col_name, col_expr in zip(spec.order_cols, spec.order_col_exprs):
            if col_expr is not None and col_name not in df.columns:
                df = df.copy()
                df[col_name] = col_expr.evaluate(df)

        # Itera por cada partição (ex: cada país)
        grupos = df.groupby(spec.partition_cols) if spec.partition_cols else [(None, df)]
        for _, grupo_df in grupos:
            # Ordena pelo critério da janela (ex: unix_date da data)
            if spec.order_cols:
                sorted_df = grupo_df.sort_values(spec.order_cols)
            else:
                sorted_df = grupo_df

            valores = sorted_df[col_name]

            # Aplica a agregação com janela deslizante
            if agg.func_name == "avg":
                if window_size:
                    agg_vals = valores.rolling(window=window_size, min_periods=1).mean()
                else:
                    agg_vals = valores.expanding().mean()
            elif agg.func_name == "sum":
                if window_size:
                    agg_vals = valores.rolling(window=window_size, min_periods=1).sum()
                else:
                    agg_vals = valores.expanding().sum()
            else:
                agg_vals = valores

            result[sorted_df.index] = agg_vals.values

        return result


# =============================================================================
# SEÇÃO 1.5 — GROUPEDDATA: resultado do groupBy()
# =============================================================================

class GroupedData:
    """
    DataFrame agrupado — análogo ao GroupedData do PySpark.
    Criado por SimulatedDataFrame.groupBy() e consumido por .agg().
    """

    def __init__(self, pdf: pd.DataFrame, group_cols: list):
        self._pdf        = pdf
        self.group_cols  = group_cols

    def agg(self, *exprs) -> "SimulatedDataFrame":
        """
        Aplica as expressões de agregação e retorna um novo SimulatedDataFrame.
        Cada AggExpr é computada de forma independente e os resultados são
        unidos pelo merge — simula o plano de execução paralelo do Spark.
        """
        pdf = self._pdf.copy()

        # Começa com as chaves de agrupamento únicas
        result = (pdf[self.group_cols]
                  .drop_duplicates()
                  .reset_index(drop=True))

        for expr in exprs:
            if not isinstance(expr, AggExpr):
                continue

            out_name = (expr._alias
                        or f"{expr.func_name}("
                           f"{expr.col._name if isinstance(expr.col, Column) else expr.col}"
                           f")")
            col = expr.col

            # Resolve a coluna: string → usa direto; Column → avalia e cria temp
            if col is None:
                col_name = "__count_star__"
                pdf[col_name] = 1
            elif isinstance(col, str):
                col_name = col
            elif isinstance(col, Column):
                col_name = f"__agg_tmp_{out_name}__"
                pdf[col_name] = col.evaluate(pdf)
            else:
                continue

            # Calcula a agregação
            grp = pdf.groupby(self.group_cols)[col_name]
            if expr.func_name == "sum":
                agg_result = grp.sum()
            elif expr.func_name in ("avg", "mean"):
                agg_result = grp.mean()
            elif expr.func_name == "max":
                agg_result = grp.max()
            elif expr.func_name == "min":
                agg_result = grp.min()
            elif expr.func_name == "count":
                agg_result = grp.count()
            elif expr.func_name == "countDistinct":
                agg_result = grp.nunique()
            elif expr.func_name == "percentile":
                p = expr.kwargs.get("p", 0.5)
                agg_result = grp.quantile(p)
            else:
                continue

            agg_df = agg_result.reset_index(name=out_name)
            result = result.merge(agg_df, on=self.group_cols, how="left")

        return SimulatedDataFrame(result)


# =============================================================================
# SEÇÃO 1.6 — SIMULATED DATAFRAME: API de alto nível
# =============================================================================

class SimulatedDataFrame:
    """
    DataFrame distribuído simulado — análogo ao pyspark.sql.DataFrame.

    Internamente usa um pandas DataFrame (_pdf) para armazenar os dados,
    mas expõe a mesma API do PySpark: filter, select, withColumn, groupBy,
    orderBy, show, printSchema, count, toPandas, cache.

    Simula o comportamento "imutável" do Spark: cada transformação retorna
    um NOVO SimulatedDataFrame, sem modificar o original.
    """

    def __init__(self, pdf: pd.DataFrame):
        self._pdf     = pdf.reset_index(drop=True)
        self._cached  = False

    # ── Transformações (retornam novo SimulatedDataFrame) ──────────────────

    def filter(self, condition) -> "SimulatedDataFrame":
        """
        Filtra linhas — equivalente ao .filter() / .where() do PySpark.
        A condição pode ser um Column booleano ou um WhenBuilder.
        """
        mask = condition.evaluate(self._pdf)
        return SimulatedDataFrame(self._pdf[mask].copy())

    def select(self, *cols) -> "SimulatedDataFrame":
        """
        Seleciona colunas — equivalente ao .select() do PySpark.
        Aceita strings (nomes de colunas) ou objetos Column.
        """
        result = {}
        for c in cols:
            if isinstance(c, str):
                result[c] = self._pdf[c]
            elif isinstance(c, Column):
                result[c._name] = c.evaluate(self._pdf)
        return SimulatedDataFrame(pd.DataFrame(result))

    def withColumn(self, name: str, col) -> "SimulatedDataFrame":
        """
        Adiciona ou substitui uma coluna — equivalente ao .withColumn() do PySpark.
        Aceita Column, WhenBuilder, ou WindowColumn.
        """
        new_pdf = self._pdf.copy()
        if isinstance(col, WindowColumn):
            new_pdf[name] = col._compute(new_pdf)
        elif isinstance(col, (Column, WhenBuilder)):
            new_pdf[name] = col.evaluate(new_pdf)
        else:
            new_pdf[name] = col
        return SimulatedDataFrame(new_pdf)

    def groupBy(self, *cols) -> GroupedData:
        """
        Agrupa o DataFrame pelas colunas especificadas —
        equivalente ao .groupBy() do PySpark. Retorna um GroupedData.
        """
        group_cols = list(cols)
        return GroupedData(self._pdf, group_cols)

    def orderBy(self, *cols, ascending=True) -> "SimulatedDataFrame":
        """
        Ordena o DataFrame — equivalente ao .orderBy() / .sort() do PySpark.
        """
        col_names = []
        asc_list  = []
        for c in cols:
            if isinstance(c, str):
                col_names.append(c)
                asc_list.append(ascending if isinstance(ascending, bool) else True)
            elif isinstance(c, Column):
                col_names.append(c._name)
                asc_list.append(True)
        sorted_pdf = self._pdf.sort_values(col_names, ascending=asc_list)
        return SimulatedDataFrame(sorted_pdf)

    def distinct(self) -> "SimulatedDataFrame":
        return SimulatedDataFrame(self._pdf.drop_duplicates())

    def cache(self) -> "SimulatedDataFrame":
        """
        No Spark, persiste o DataFrame em memória para reutilização.
        Nesta simulação é um no-op — pandas já mantém tudo em memória.
        """
        self._cached = True
        print("  [cache] DataFrame marcado como cached (no-op na simulação)")
        return self

    # ── Ações (executam e retornam resultado) ──────────────────────────────

    def count(self) -> int:
        """Conta o número de linhas — equivalente ao .count() do PySpark."""
        return len(self._pdf)

    def show(self, n: int = 20, truncate: bool = True) -> None:
        """
        Exibe as primeiras n linhas — equivalente ao .show() do PySpark.
        """
        pdf = self._pdf.head(n)
        col_width = 20 if truncate else 40
        header = " | ".join(f"{c[:col_width]:<{col_width}}" for c in pdf.columns)
        sep    = "-+-".join("-" * col_width for _ in pdf.columns)
        print(f"+{sep}+")
        print(f"|{header}|")
        print(f"+{sep}+")
        for _, row in pdf.iterrows():
            vals = " | ".join(
                f"{str(v)[:col_width]:<{col_width}}" for v in row
            )
            print(f"|{vals}|")
        print(f"+{sep}+")
        print(f"apenas exibindo {min(n, len(self._pdf))} de {len(self._pdf)} linhas\n")

    def printSchema(self) -> None:
        """Exibe o schema (tipos das colunas) — equivalente ao .printSchema() do PySpark."""
        print("root")
        for col, dtype in self._pdf.dtypes.items():
            print(f" |-- {col}: {dtype} (nullable = true)")

    def collect(self) -> list:
        """Retorna todas as linhas como lista de dicionários."""
        return self._pdf.to_dict(orient="records")

    def agg(self, *exprs) -> "SimulatedDataFrame":
        """Agrega todo o DataFrame sem agrupamento."""
        result = {}
        for expr in exprs:
            if not isinstance(expr, AggExpr):
                continue
            out_name = expr._alias or expr.func_name
            col = expr.col
            if isinstance(col, str):
                s = self._pdf[col]
            elif isinstance(col, Column):
                s = col.evaluate(self._pdf)
            else:
                continue
            if expr.func_name in ("min", "max"):
                result[out_name] = [getattr(s, expr.func_name)()]
            elif expr.func_name == "countDistinct":
                result[out_name] = [s.nunique()]
            elif expr.func_name == "count":
                result[out_name] = [s.count()]
        return SimulatedDataFrame(pd.DataFrame(result))

    def toPandas(self) -> pd.DataFrame:
        """Converte para pandas DataFrame — equivalente ao .toPandas() do PySpark."""
        return self._pdf.copy()

    def __repr__(self):
        return (f"SimulatedDataFrame[{len(self._pdf)} linhas × "
                f"{len(self._pdf.columns)} colunas]")


# =============================================================================
# SEÇÃO 1.7 — RDD: processamento de baixo nível (MapReduce)
# =============================================================================

class RDD:
    """
    Resilient Distributed Dataset (RDD) simulado.

    No Spark real, um RDD é uma coleção imutável de dados distribuída entre
    os nós do cluster em partições. Aqui simulamos esse comportamento
    dividindo os dados em listas Python (as "partições").

    Operações suportadas:
      .map(func)         — transforma cada elemento (fase MAP)
      .filter(func)      — filtra elementos por condição
      .flatMap(func)     — map + achatamento da lista resultante
      .reduceByKey(func) — agrega por chave (fase REDUCE / shuffle)
      .collect()         — coleta todos os dados das partições
      .count()           — conta o total de elementos
      .take(n)           — retorna os primeiros n elementos
    """

    def __init__(self, data: list, num_partitions: int = 4):
        """
        Cria o RDD dividindo 'data' em 'num_partitions' partições.
        Simula a distribuição de dados entre nós do cluster.
        """
        n    = len(data)
        size = max(1, math.ceil(n / num_partitions))
        self.partitions     = [data[i * size:(i + 1) * size]
                                for i in range(num_partitions)]
        self.num_partitions = num_partitions

    def map(self, func) -> "RDD":
        """
        FASE MAP — aplica func a cada elemento de cada partição.
        Simula o processamento paralelo: cada partição seria processada
        por um executor diferente no cluster real.
        """
        novos = []
        for particao in self.partitions:
            novos.extend([func(item) for item in particao])
        return RDD(novos, self.num_partitions)

    def filter(self, func) -> "RDD":
        """Filtra elementos que satisfazem a condição em cada partição."""
        filtrados = []
        for particao in self.partitions:
            filtrados.extend([item for item in particao if func(item)])
        return RDD(filtrados, self.num_partitions)

    def flatMap(self, func) -> "RDD":
        """map + achata listas aninhadas — útil para tokenização, por exemplo."""
        resultado = []
        for particao in self.partitions:
            for item in particao:
                resultado.extend(func(item))
        return RDD(resultado, self.num_partitions)

    def reduceByKey(self, func) -> dict:
        """
        FASE REDUCE — agrupa por chave e aplica a função de redução.

        Simula o fluxo completo do MapReduce:
          1. SHUFFLE: redistribui os pares (chave, valor) por chave
          2. SORT   : ordena por chave dentro de cada grupo
          3. REDUCE : aplica func para colapsar os valores de cada chave

        No Spark real, o shuffle move dados entre executores pela rede —
        a operação mais custosa do MapReduce.
        """
        # 1. Shuffle: agrupa os valores por chave
        grupos = defaultdict(list)
        for particao in self.partitions:
            for chave, valor in particao:
                grupos[chave].append(valor)

        # 2. Reduce: aplica a função de agregação
        return {chave: func(valores) for chave, valores in grupos.items()}

    def collect(self) -> list:
        """Coleta todos os elementos de todas as partições."""
        resultado = []
        for particao in self.partitions:
            resultado.extend(particao)
        return resultado

    def count(self) -> int:
        """Conta o total de elementos em todas as partições."""
        return sum(len(p) for p in self.partitions)

    def take(self, n: int) -> list:
        """Retorna os primeiros n elementos (sem coletar tudo em memória)."""
        resultado = []
        for particao in self.partitions:
            for item in particao:
                resultado.append(item)
                if len(resultado) == n:
                    return resultado
        return resultado

    def __repr__(self):
        return f"SimulatedRDD[{self.count()} registros em {self.num_partitions} partições]"


# =============================================================================
# SEÇÃO 1.8 — SPARKCONTEXT e SPARKSESSION
# =============================================================================

class SimulatedSparkContext:
    """
    SparkContext simulado — ponto de entrada para operações de baixo nível (RDD).
    No Spark real gerencia a conexão com o cluster e o agendamento de jobs.
    """

    def __init__(self, master: str = "local[*]", app_name: str = "SimulatedSpark"):
        self.master   = master
        self.app_name = app_name
        print(f"  SparkContext iniciado: master={master}, app={app_name}")

    def parallelize(self, data: list, num_slices: int = 4) -> RDD:
        """
        Distribui uma coleção Python em um RDD com num_slices partições.
        Equivalente ao sc.parallelize() do PySpark.
        """
        return RDD(data, num_slices)

    def setLogLevel(self, level: str) -> None:
        pass  # no-op na simulação


class SimulatedSparkSession:
    """
    SparkSession simulada — ponto de entrada unificado para toda aplicação Spark.

    No Spark real, a SparkSession encapsula SparkContext, SQLContext e
    HiveContext. Aqui ela gerencia a leitura de dados (via pandas) e expõe
    o SparkContext para operações RDD.

    Uso idêntico ao PySpark:
        spark = SimulatedSparkSession.builder \\
            .appName("MeuApp") \\
            .master("local[*]") \\
            .getOrCreate()
        df = spark.read.csv("dados.csv")
    """

    class _Builder:
        def __init__(self):
            self._app_name = "SimulatedSpark"
            self._master   = "local[*]"

        def appName(self, name: str) -> "SimulatedSparkSession._Builder":
            self._app_name = name
            return self

        def master(self, m: str) -> "SimulatedSparkSession._Builder":
            self._master = m
            return self

        def config(self, key: str, value) -> "SimulatedSparkSession._Builder":
            return self  # ignora configs na simulação

        def getOrCreate(self) -> "SimulatedSparkSession":
            return SimulatedSparkSession(self._app_name, self._master)

    builder = _Builder()

    def __init__(self, app_name: str, master: str):
        self.sparkContext = SimulatedSparkContext(master, app_name)
        self.version      = "4.1.1-simulado"
        print(f"\n{'='*60}")
        print(f"  SimulatedSparkSession iniciada")
        print(f"  Versão  : {self.version}")
        print(f"  Master  : {master}")
        print(f"  App     : {app_name}")
        print(f"{'='*60}\n")

    class _Reader:
        def __init__(self, session):
            self._session = session

        def csv(self, path: str, header: bool = True,
                inferSchema: bool = True) -> SimulatedDataFrame:
            """
            Lê um CSV com pandas e retorna um SimulatedDataFrame.
            Simula o spark.read.csv() — em um cluster real, o Spark
            leria diferentes blocos do arquivo em diferentes executores.
            """
            pdf = pd.read_csv(path)
            return SimulatedDataFrame(pdf)

    @property
    def read(self) -> "_Reader":
        return self._Reader(self)

    def stop(self) -> None:
        print("\nSimulatedSparkSession encerrada.\n")


# =============================================================================
# SEÇÃO 1.9 — F: namespace de funções (espelho de pyspark.sql.functions)
# =============================================================================

class F:
    """
    Funções de coluna — espelho simplificado de pyspark.sql.functions.
    Cada método retorna um Column (ou AggExpr) que encapsula a operação.
    """

    @staticmethod
    def col(name: str) -> Column:
        """Referência a uma coluna pelo nome."""
        return Column(lambda df, _n=name: df[_n], name)

    @staticmethod
    def lit(value) -> Column:
        """Valor literal (constante)."""
        return Column(lambda df, _v=value: pd.Series([_v] * len(df), index=df.index), str(value))

    # ── Funções de data ──────────────────────────────────────────────────────

    @staticmethod
    def to_date(col, fmt: str = None) -> Column:
        col = _to_col(col)
        def _eval(df, _c=col):
            return pd.to_datetime(_c.evaluate(df), format=fmt, errors="coerce")
        return Column(_eval, f"to_date({col._name})")

    @staticmethod
    def year(col) -> Column:
        col = _to_col(col)
        return Column(lambda df, _c=col: _c.evaluate(df).dt.year,
                      f"year({col._name})")

    @staticmethod
    def date_format(col, fmt: str) -> Column:
        """Formata data para string — converte formato Java (yyyy-MM) para Python (%Y-%m)."""
        col  = _to_col(col)
        py_fmt = (fmt.replace("yyyy", "%Y").replace("MM", "%m")
                      .replace("dd", "%d").replace("HH", "%H"))
        def _eval(df, _c=col, _f=py_fmt):
            s = _c.evaluate(df)
            if hasattr(s, "dt"):
                return s.dt.strftime(_f)
            return pd.to_datetime(s, errors="coerce").dt.strftime(_f)
        return Column(_eval, f"date_format({col._name}, {fmt})")

    @staticmethod
    def unix_date(col) -> Column:
        """Dias desde 1970-01-01 — substituto de cast('long') no Spark 4.x."""
        col = _to_col(col)
        epoch = pd.Timestamp("1970-01-01")
        def _eval(df, _c=col, _e=epoch):
            s = _c.evaluate(df)
            if not pd.api.types.is_datetime64_any_dtype(s):
                s = pd.to_datetime(s, errors="coerce")
            return (s - _e).dt.days
        return Column(_eval, f"unix_date({col._name})")

    # ── Funções de controle de fluxo ─────────────────────────────────────────

    @staticmethod
    def when(condition, value) -> WhenBuilder:
        """Inicia uma expressão condicional encadeada."""
        return WhenBuilder(condition, value)

    # ── Funções de agregação ─────────────────────────────────────────────────

    @staticmethod
    def sum(col) -> AggExpr:
        return AggExpr("sum", _to_col(col) if not isinstance(col, str) else col)

    @staticmethod
    def avg(col) -> AggExpr:
        return AggExpr("avg", _to_col(col) if not isinstance(col, str) else col)

    @staticmethod
    def max(col) -> AggExpr:
        return AggExpr("max", _to_col(col) if not isinstance(col, str) else col)

    @staticmethod
    def min(col) -> AggExpr:
        return AggExpr("min", _to_col(col) if not isinstance(col, str) else col)

    @staticmethod
    def count(col="*") -> AggExpr:
        if col == "*":
            return AggExpr("count", None)
        return AggExpr("count", _to_col(col) if not isinstance(col, str) else col)

    @staticmethod
    def countDistinct(col) -> AggExpr:
        return AggExpr("countDistinct", _to_col(col) if not isinstance(col, str) else col)

    @staticmethod
    def percentile_approx(col, p: float) -> AggExpr:
        return AggExpr("percentile", _to_col(col) if not isinstance(col, str) else col, p=p)

    # ── Funções escalares extras ─────────────────────────────────────────────

    @staticmethod
    def abs(col) -> Column:
        col = _to_col(col)
        return Column(lambda df, _c=col: _c.evaluate(df).abs(), f"abs({col._name})")


# Atalho para usar AggExpr.over(window_spec) — igual ao PySpark
def _agg_over(self, window_spec: WindowSpec) -> WindowColumn:
    return WindowColumn(self, window_spec)

AggExpr.over = _agg_over


# =============================================================================
# PARTE 2 — PIPELINE ETL APLICADO AO DATASET COVID-19
# =============================================================================

# Regiões que o OWID inclui como agregações (não são países individuais)
REGIOES_AGREGADAS = [
    "World", "Asia", "Europe", "Africa", "North America",
    "South America", "Oceania", "European Union",
    "High income", "Low income", "Lower middle income",
    "Upper middle income", "High-income countries",
]


def criar_sessao() -> SimulatedSparkSession:
    """Inicializa a SparkSession simulada."""
    return (
        SimulatedSparkSession.builder
        .appName("EDA_COVID19_Simulado")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def extrair(spark: SimulatedSparkSession, caminho: str) -> SimulatedDataFrame:
    """
    EXTRACT — lê o CSV e converte a coluna date para DateType.
    Simula o spark.read.option("header","true").inferSchema(true).csv().
    """
    df = spark.read.csv(caminho)
    # Converte date para datetime e colunas numéricas
    pdf = df._pdf.copy()
    pdf["date"] = pd.to_datetime(pdf["date"], errors="coerce")
    for col in ["new_cases", "new_deaths", "total_cases", "total_deaths",
                "population", "new_vaccinations", "people_fully_vaccinated",
                "reproduction_rate", "stringency_index", "gdp_per_capita",
                "median_age", "life_expectancy"]:
        if col in pdf.columns:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
    return SimulatedDataFrame(pdf)


def exibir_visao_geral(df: SimulatedDataFrame) -> None:
    """Etapa 1 — exibe estatísticas básicas do dataset bruto."""
    pdf = df._pdf
    print("=" * 60)
    print("ETAPA 1 — Visão Geral do Dataset")
    print("=" * 60)
    print(f"  Linhas      : {df.count():,}")
    print(f"  Colunas     : {len(pdf.columns)}")
    print(f"  Países      : {pdf['country'].nunique()}")
    print(f"  Período     : {pdf['date'].min().date()} → {pdf['date'].max().date()}")
    print(f"  Continentes : {pdf['continent'].dropna().nunique()}")
    print()
    df.printSchema()


def demonstrar_rdd(df: SimulatedDataFrame, spark: SimulatedSparkSession) -> None:
    """
    Etapa 2 — demonstra o modelo RDD (baixo nível) com MapReduce.
    Mostra as fases MAP, SHUFFLE e REDUCE aplicadas ao dataset de COVID.
    """
    print("=" * 60)
    print("ETAPA 2 — Pipeline MapReduce (RDD — baixo nível)")
    print("=" * 60)

    # Converte o DataFrame para lista de registros (simula RDD de linhas)
    pdf_amostra = df._pdf[df._pdf["continent"].notna()].head(10_000)
    registros   = pdf_amostra.to_dict(orient="records")
    rdd_covid   = spark.sparkContext.parallelize(registros, num_slices=4)
    print(f"\n  RDD criado: {rdd_covid}")

    # FASE MAP: extrai (país, novos_casos) de cada registro
    print("\n  [MAP] Extraindo pares (país, novos_casos)...")
    rdd_pares = rdd_covid.map(
        lambda r: (r["country"], r["new_cases"] if pd.notna(r["new_cases"]) else 0)
    )
    print(f"  Exemplo de pares: {rdd_pares.take(3)}")

    # FASE REDUCE: soma casos por país
    print("\n  [SHUFFLE + REDUCE] Somando casos por país...")
    total_por_pais = rdd_pares.reduceByKey(lambda valores: sum(valores))
    top5 = sorted(total_por_pais.items(), key=lambda x: -x[1])[:5]
    for pais, total in top5:
        print(f"    {pais:<20}: {total:>12,.0f} casos")

    # FILTER: dias com mais de 10.000 casos
    print("\n  [FILTER] Dias com mais de 10.000 novos casos...")
    rdd_alto = rdd_covid.filter(
        lambda r: pd.notna(r["new_cases"]) and r["new_cases"] > 10_000
    )
    print(f"  {rdd_alto.count():,} registros (de {rdd_covid.count():,} totais)\n")


def analisar_nulos(df: SimulatedDataFrame) -> None:
    """Etapa 3 — calcula e exibe a proporção de nulos nas colunas principais."""
    pdf = df._pdf
    total = len(pdf)
    colunas = [
        "total_cases", "new_cases", "total_deaths", "new_deaths",
        "people_fully_vaccinated", "hosp_patients", "icu_patients",
        "stringency_index", "reproduction_rate",
    ]
    print("=" * 60)
    print("ETAPA 3 — Qualidade dos Dados (valores nulos)")
    print("=" * 60)
    for col in colunas:
        if col in pdf.columns:
            n = pdf[col].isna().sum()
            print(f"  {col:<40}: {n:>8,}  ({n/total*100:5.1f}%)")
    print()


def transformar(df: SimulatedDataFrame) -> SimulatedDataFrame:
    """
    TRANSFORM — limpeza e enriquecimento do dataset.
    Aplica filtros e cria colunas derivadas usando a API DataFrame simulada.
    """
    print("=" * 60)
    print("ETAPA 4 — Transformação (DataFrame API — alto nível)")
    print("=" * 60)

    # Remove regiões agregadas (mantém só países)
    df_paises = df.filter(F.col("continent").isNotNull())
    df_paises = df_paises.filter(~F.col("country").isin(*REGIOES_AGREGADAS))
    print(f"  Após remover regiões : {df_paises.count():,}")

    # Remove negativos
    df_limpo = df_paises.filter(
        F.col("new_cases").isNull().__or__(F.col("new_cases") >= 0)
    )
    df_limpo = df_limpo.filter(
        F.col("new_deaths").isNull().__or__(F.col("new_deaths") >= 0)
    )
    print(f"  Após remover neg.    : {df_limpo.count():,}")

    # Colunas derivadas
    df_enr = (
        df_limpo
        .withColumn("year_month",
                    F.date_format(F.col("date"), "yyyy-MM"))
        .withColumn("year",
                    F.year(F.col("date")))
        .withColumn("daily_mortality_rate",
                    F.when(
                        (F.col("new_cases") > 0) & F.col("new_deaths").isNotNull(),
                        F.col("new_deaths") / F.col("new_cases") * 100
                    ).otherwise(F.lit(None)))
        .withColumn("fully_vaccinated_pct",
                    F.when(
                        (F.col("population") > 0) & F.col("people_fully_vaccinated").isNotNull(),
                        F.col("people_fully_vaccinated") / F.col("population") * 100
                    ).otherwise(F.lit(None)))
    )
    print("  Colunas adicionadas  : year_month, year, daily_mortality_rate, "
          "fully_vaccinated_pct\n")
    return df_enr


def adicionar_media_movel(df: SimulatedDataFrame,
                          coluna: str = "new_cases",
                          janela: int = 7) -> SimulatedDataFrame:
    """
    WINDOW FUNCTION — média móvel por país.
    Simula o Window.partitionBy("country").orderBy("date").rowsBetween(-6,0)
    do PySpark real.
    """
    nome_saida  = f"{coluna}_ma{janela}"
    # Ordena por unix_date para garantir ordering numérico correto
    window_spec = (
        Window
        .partitionBy("country")
        .orderBy(F.unix_date(F.col("date")))
        .rowsBetween(-(janela - 1), 0)
    )
    agg_over = F.avg(coluna).over(window_spec)
    return df.withColumn(nome_saida, agg_over)


def agregar_por_pais(df: SimulatedDataFrame) -> SimulatedDataFrame:
    """agrega métricas totais por país."""
    return (
        df
        .groupBy("country", "continent", "population",
                 "gdp_per_capita", "median_age", "life_expectancy")
        .agg(
            F.max("total_cases").alias("total_cases"),
            F.max("total_deaths").alias("total_deaths"),
            F.max("people_fully_vaccinated").alias("people_fully_vaccinated"),
            F.avg("stringency_index").alias("avg_stringency_index"),
            F.avg("reproduction_rate").alias("avg_reproduction_rate"),
        )
    )


def agregar_mensal_global(df: SimulatedDataFrame) -> SimulatedDataFrame:
    """Agrega novos casos e mortes por mês globalmente."""
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


def agregar_por_continente(df: SimulatedDataFrame) -> SimulatedDataFrame:
    """Agrega totais por continente."""
    return (
        df
        .groupBy("continent")
        .agg(
            F.max("total_cases").alias("total_cases"),
            F.max("total_deaths").alias("total_deaths"),
            F.avg("fully_vaccinated_pct").alias("avg_fully_vaccinated_pct"),
            F.avg("stringency_index").alias("avg_stringency_index"),
        )
        .orderBy("total_cases")
    )


def salvar(df: SimulatedDataFrame, nome: str) -> None:
    """LOAD — salva o resultado em CSV."""
    os.makedirs(CAMINHO_SAIDA, exist_ok=True)
    destino = os.path.join(CAMINHO_SAIDA, f"{nome}.csv")
    df._pdf.to_csv(destino, index=False)
    print(f"  Salvo: {destino}")



#                                   execução prévia
#                               depois simulacao.ipynb


if __name__ == "__main__":

    spark = criar_sessao()

    # extração
    df_bruto = extrair(spark, CAMINHO_CSV)
    exibir_visao_geral(df_bruto)

    # ── RDD / MapReduce ────────────────────────────────────────────────────
    demonstrar_rdd(df_bruto, spark)
    analisar_nulos(df_bruto)

    # transformação
    df_limpo = transformar(df_bruto)

    print("=" * 60)
    print("ETAPA 5 — Window Function (Média Móvel 7 dias)")
    print("=" * 60)
    df_completo = adicionar_media_movel(df_limpo, "new_cases",  7)
    df_completo = adicionar_media_movel(df_completo, "new_deaths", 7)

    print("  Amostra — Brasil, últimas 5 linhas:")
    brasil = (df_completo._pdf[df_completo._pdf["country"] == "Brazil"]
              [["date", "new_cases", "new_cases_ma7", "new_deaths", "new_deaths_ma7"]]
              .dropna(subset=["new_cases"])
              .tail(5))
    print(brasil.to_string(index=False))
    print()

    # agregação
    print("=" * 60)
    print("ETAPA 6 — Agregações (groupBy + agg)")
    print("=" * 60)

    df_pais = agregar_por_pais(df_completo)
    print("\nTop 10 países por total de casos:")
    top10 = (df_pais._pdf
             .dropna(subset=["total_cases"])
             .nlargest(10, "total_cases")
             [["country", "continent", "total_cases", "total_deaths"]])
    print(top10.to_string(index=False))

    print("\nTotais por continente:")
    df_cont = agregar_por_continente(df_completo)
    print(df_cont._pdf.to_string(index=False))

    # ── LOAD ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ETAPA 7 — Carga (Load)")
    print("=" * 60)
    salvar(df_pais,                     "resumo_por_pais_sim")
    salvar(agregar_mensal_global(df_completo), "evolucao_mensal_sim")
    salvar(agregar_por_continente(df_completo), "resumo_continente_sim")

    print("\n" + "=" * 60)
    print("Pipeline simulado concluído com sucesso!")
    print("=" * 60)

    spark.stop()
