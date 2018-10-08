from sklearn.datasets import load_boston

boston = load_boston()
#itens da base
boston.keys()

print(boston.DESCR)

import pandas

tabela = pandas.DataFrame(boston.data)
tabela.columns = boston.feature_names
tabela.head()