from sklearn.datasets import load_diabetes
import pandas

diabetes = load_diabetes()
#itens da base
diabetes.keys()
print(diabetes.DESCR)

tabela = pandas.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names
tabela.head()
tabela.head(10)
tabela['Preço'] = diabetes.target
tabela.head(10)
import matplotlib.pyplot as plt
# qual a melhor característica/columa que melhor representa o preço?

#visualmente
plt.scatter(tabela.age, tabela.Preço)
plt.xlabel('Idade')
plt.ylabel('Preço')
plt.show()

#visualmente
plt.scatter(tabela.sex, tabela.Preço)
plt.xlabel('idade')
plt.ylabel('Preço')
plt.show()

#visualmente
plt.scatter(tabela.bmi, tabela.Preço)
plt.xlabel('Indice de massa corporea')
plt.ylabel('Preço')
plt.show()

#visualmente
plt.scatter(tabela.bp, tabela.Preço)
plt.xlabel('Pressao Sanguinea')
plt.ylabel('Preço')
plt.show()

tabela.corr()

plt.scatter(tabela.s2, tabela.Preço)
plt.xlabel('% lower status')
plt.ylabel('Preço')
plt.show()

from sklearn.datasets import load_boston
import pandas

diabetes = load_diabetes()
tabela = pandas.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names

#seleciona duas colunas
X = tabela[["bmi", "s2"]]
print(X)