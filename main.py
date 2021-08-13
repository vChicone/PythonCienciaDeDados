import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

#Importar arquivo
tabela = pd.read_csv("advertising.csv")
display(tabela)
print(tabela.info())

#Análise exploratória
sns.heatmap(tabela.corr(), cmap="Wistia", annot=True)
plt.show() #Correlação < 0,7
sns.pairplot(tabela)
plt.show()

#Modelagem dos dados
#Definir x e y
y = tabela["Vendas"] #y=quem eu quero descobrir
x = tabela.drop("Vendas", axis=1)
#Aplicar o teste
x_treino, x_test, y_treino, y_test = train_test_split(x, y) #tamanho do test (,test_size=0.3)

#Aplicar AI
modeloRegressaoLinear = LinearRegression()
modeloRandomForest = RandomForestRegressor()
modeloRegressaoLinear.fit(x_treino, y_treino)
modeloRandomForest.fit(x_treino, y_treino)

#Teste AI
previsaoRegressaoLinear = modeloRegressaoLinear.predict(x_test)
previsaoRandomForest = modeloRandomForest.predict(x_test)
#R2 -> 0 a 100%, quanto maior o valor, melhor
print(metrics.r2_score(y_test, previsaoRegressaoLinear))
print(metrics.r2_score(y_test, previsaoRandomForest))

#Viasualizar as previsões
tabelaAuxiliar = pd.DataFrame()
tabelaAuxiliar["y_teste"] = y_test
tabelaAuxiliar["Regressão Linear"] = previsaoRegressaoLinear
tabelaAuxiliar["Random Forest"] = previsaoRandomForest
plt.figure(figsize=(15, 5))
sns.lineplot(data=tabelaAuxiliar)
plt.show()
sns.barplot(x=x_treino.columns, y=modeloRandomForest.feature_importances_)
plt.show()

#Calcular previsão
tabelaPrevisao = pd.read_csv("tabelaprevisao.csv")
previsao = modeloRandomForest.predict(tabelaPrevisao)
print(previsao)

