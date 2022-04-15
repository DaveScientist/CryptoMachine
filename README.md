# CryptoMachine
This is my new project of Machine Learning made for the school "The Bridge".

<img src="https://i0.wp.com/sistemasgeniales.com/wp-content/uploads/2020/08/CRYPTOBIGDATA2.jpg?resize=696%2C463&ssl=1" alt="drawing" width="600"/>


### INTRODUCCIÓN

La Inteligencia Artificial (IA) ha supuesto un gran impacto tecnológico en numerosos ámbitos de la sociedad. Se trata de una forma de computación y de transformación de la información.

El aprendizaje automático o Machine Learning es una rama de la IA que se basa en el aprendizaje de las computadoras con la mínima interacción humana.
A su vez, existe otra rama dentro del Machine Learning, el aprendizaje supervisado, el cual se centra en que las máquinas aprendan por sí mismas a través del entrenamiento 
y dados unos datos de entrada se obtienen unos datos de salida.

En el presente proyecto para la escuela "The Bridge" se aplican técnicas de aprendizaje supervisado con el objetivo de hacer predicciones de los valores de las criptomonedas,
tratando de reducir la alta volatilidad a la que se ve expuesto el mercado.


Como ya hicimos anteriormente en el EDA, seguiremos hablando de las criptomonedas aunque en este caso,
vamos a tratar de crear un modelo de Machine Learning para intentar predecir el comportamiento del precio de las principales criptomonedas.


**- Bitcoin**
**- Cardano**
**- Ethereum**


### ÍNDICE


1. [ANÁLISIS EXPLORATORIO](#ANÁLISIS_EXPLORATORIO)
2. [MODELO DE REGRESIÓN LINEAL](#MODELO_DE_REGRESIÓN_LINEAL)
3. [MÉTRICAS DE REGRESIÓN](#MÉTRICAS_DE_REGRESIÓN)
4. [TÉCNICA DE RANDOM FOREST](#TÉCNICA_DE_RAMDOM_FOREST)
5. [TRAIN Y TEST](#TRAIN_Y_TEST)




### 1. ANÁLISIS EXPLORATORIO <a id='ANÁLISIS_EXPLORATORIO'></a>

Lo primero de todo lo que haremos, será descargar todos los datos de los precios de las criptomonedas anteriormente señaladas
para, a través de los datasets facilitados por Yahoo Finance, analizar los precios en un rango de tiempo determinado y su posterior representación gráfica 
para entender mejor la volatilidad y las diferentes variables a la que nos encontramos dentro del dataset.


### 2. MODELO DE REGRESIÓN LINEAL <a id='MODELO_DE_REGRESIÓN_LINEAL'></a>

Vamos a realizar nuestro primer modelo de predicción una vez hemos limpiado todos los datos del dataset y visualizado correlaciones y variables numéricas.

El primer modelo es el de regresión lineal ya que debido al gran conocimiento adquirido dentro del mundo del trading, de las finanzas y del mercado financiero, 
creo que podría ser el modelo que mejor se ajusta para poder hacer un modelo predictivo dentro del aprendizaje supervisado en Machine Learning.


### 3. MÉTRICAS DE REGRESIÓN <a id='MÉTRICAS_DE_REGRESIÓN'></a>

La evaluación de cada modelo se realizará en un conjunto de validación mediante la raíz del error cuadrático medio (RMSE), el error absoluto medio (MAE),
el error porcentual absoluto medio (MAPE) y el coeficiente de determinación (R2).


### 4. TÉCNICA DE RANDOM FOREST <a id='TÉCNICA_DE_RANDOM_FOREST'></a>

El procedimiento que se ha seguido para realizar este estudio consta de varias etapas. 
Se han obtenido datos históricos y se han construido modelos de predicción utilizando Random Forest que predecirá los precios de cierre del conjunto de prueba.
Se realizará una comparación entre los modelos propuestos en función de la cantidad de información obtenida de cada criptomoneda. 


### 5. TRAIN Y TEST <a id='TRAIN_Y_TEST'></a>

Para comprobar la eficacia de cada modelo propuesto se crearán un conjunto de entrenamiento y otro de prueba, siendo este último de menor tamaño. De esta forma, el 80% más antiguos de los datos serán 
los datos de entrenamiento y el 20% restante más reciente corresponderá a los datos de prueba.


<img src="https://imagenes.elpais.com/resizer/G9srcPjblTzItvubVDaS-jez3To=/980x0/cloudfront-eu-central-1.images.arcpublishing.com/prisa/OGO3RZFJXFGYFJPN2SYW6HO4BI.jpg" alt="drawing" width="600"/>


