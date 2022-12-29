
"""
Sección: Gráficos
"""
def graficar_nulos_msno(df):
    """
    Función: Grafica de los nulos de todo el dataframe.
    Input: Dataframe con la info.
    Output: Gráfica con la librería msno.
    """
    
    import missingno as msno

    msno.matrix(df)


def graficar_mapa_calor_msno(df):
    """
    Función: Grafica un mapa de calor de las variables del dataframe.
    Input: Dataframe con la info.
    Output: Gráfica de mapa de calor con la librería msno.
    """
    
    import missingno as msno

    msno.heatmap(df)


def graficar_porcentajes_nulos(nulos):
    """
    Función: Grafica en barras la serie recibida con los nulos.
    Input: Serie con los datos de los nulos a graficar.
    Output: Gráfica de barras.
    """

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    nulos.plot(kind='bar')


def graficar_outliers(df, outliers=True):
    """
    Función: Grafica boxplot con o sin atípicos del dataframe según el valor del 
    parámetro enviado.
    Input: Dataframe con la info y Booleano indicando si quiere o no con outliers.
    Output: Gráfica de boxplot.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(14, 6))
    if(outliers):
        titulo = 'Variables numéricas con atípicos'
        color = 'green'
    else:
        titulo = 'Variables numéricas sin atípicos'
        color = 'turquoise'
    sns.boxplot(data=df, showfliers=outliers, color=color).set(title=titulo)  

def graficar_distribucion_datos(data, cantidad_barras=7):
    """
    Función: Grafica un histograma de la columna pasada como parámetro.
    Input: Serie con la columna a graficar.
    Output: Gráfica de histograma.
    """
    
    import matplotlib.pyplot as plt

    plt.hist(data, bins=cantidad_barras)  


def formatear_autopct(data):
    """
    Función: Permite agregar al piechart, info de la cantidad en número de cada porción 
    del piechart, no solo los porcentajes. Para que no dé error la función se debe 
    llamar my_format y debe recibir pct como argumento por defecto.
    Input: Serie con la columna a graficar.
    Output: Serie con los datos formateados.
    """

    def my_format(pct):
        total = sum(data)        
        val = int(round(pct*total/100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)

    return my_format


"""
Sección: Cálculos
"""
def calcular_mad(data, axis=None):
    """
    Función: Calcula la desviación absoluta media de una columna.
    Input: Serie con la columan a calcular.
    Output: Número flotante.
    """

    from numpy import mean, absolute

    return mean(absolute(data - mean(data, axis)), axis)


def calcular_porcentajes_nulos(df):
    """
    Función: Calcula el porcentaje de nulos de un df.
    Input: Dataframe.
    Ouput: Serie con los porcentajes de nulos.
    """

    porcen_nulos = (df.isnull().sum()/df.shape[0])*100
    porcen_nulos = porcen_nulos.sort_values(ascending=False)
    porcen_nulos = porcen_nulos[porcen_nulos>0]

    return porcen_nulos


def calcular_medidas_tendencia_central(data):
    """
    Función: Para un columna numérica calcula: resumen estadístico, media geométrica,
    media armónica, media recortada y la moda.
    Input: Serie con la columna a realizar los cálculos.
    Ouput: Impresión de los cálculos.
    """

    from scipy import stats
    import warnings

    # Deshabilito los warnings
    warnings.filterwarnings('ignore')

    print()
    print("Impresión de Medidas de Tendencia Central")
    print("-----------------------------------------")
    print(f"Resumen: {stats.describe(data)}")
    print(f"Media geométrica: {stats.gmean(data)}")
    print(f"Media armónica: {stats.hmean(data)}")
    print(f"Media Recortada: {stats.trim_mean(data, 0.1)}") # Proporción removida en cada cola 10%
    print(f"Moda: {stats.mode(data)}")
    

def calcular_medidas_localizacion(data):
    """
    Función: Para un columna numérica calcula: percentiles, cuartiles y deciles.
    Input: Serie con la columna a realizar los cálculos.
    Ouput: Impresión de los cálculos.
    """

    import numpy as np

    print()
    print("Impresión de Medidas de Localización")
    print("------------------------------------")
    print(f"Percentiles: {np.percentile(data, [25, 75, 90])}")
    print(f"Cuartiles: {np.percentile(data, [0, 25, 75, 100])}")
    print(f"Deciles: {np.percentile(data, [np.arange(0, 100, 10)])}")
  

def calcular_medidas_dispercion_absolutas(data):
    """
    Función: Para un columna numérica calcula: varianza, desviación standard, rango 
    intercuartílico, desviación de cuartíl y desviación absoluta media(MAD).
    Input: Serie con la columna a realizar los cálculos.
    Ouput: Impresión de los cálculos.
    """

    import numpy as np
    from scipy import stats

    print()
    print("Impresión de Medidas de Disperción Absolutas")
    print("--------------------------------------------")
    print(f"Varianza: {stats.describe(data)[3]}")
    print(f"Desvicaión standard: {np.sqrt(stats.describe(data)[3])}")
    print(f"Rango intercuartílico: {stats.iqr(data) }")
    print(f"Desviación del cuartíl: {(np.percentile(data, 75)-np.percentile(data, 25))/2}")
    print(f"Desviación absoluta media(MAD): {calcular_mad(data)}")
  

def calcular_medidas_dispercion_relativas(data):
    """
    Función: Para un columna numérica calcula: coeficiente de variación(CV), 
    coeficiente de variación de la desviación del cuartíl y error estandar.
    Input: Serie con la columna a realizar los cálculos.
    Ouput: Impresión de los cálculos.
    """

    import numpy as np
    from scipy import stats

    print()
    print("Impresión de Medidas de Disperción Relativas")
    print("--------------------------------------------")
    print(f"Coeficiente de variación(CV): {stats.variation(data)}")
    print(f"Coeficiente de variación de la desvicación del cuartíl: {(np.percentile(data, 75)-np.percentile(data, 25))/(np.percentile(data, 75)+np.percentile(data, 25))}")
    print(f"Erro standard: {stats.sem(data)}")
    
  
def calcular_medidas_asimetria_curtosis(data):
    """
    Función: Para un columna numérica calcula: asimetría y curtosis.
    Input: Serie con la columna a realizar los cálculos.
    Ouput: Impresión de los cálculos.
    """

    from scipy import stats

    print()
    print("Impresión de Medidas de Asimetría y Curtosis")
    print("--------------------------------------------")
    print(f"Asimetría: {stats.skew(data)}")
    print(f"Curtosis: {stats.kurtosis(data)-3}")
    
  
"""
Sección: Imputación
"""
def imputar_outliers_IQR(df):
    """
    Función: Permite aplicar el criterio de reemplazo de atípicos. Se reemplaza el
    atípico por la mediana que es una medida robusta(no influenciada por outliers). 
    Input: Dataframe con las columnas a imputar.
    Output: Dataframe con las columnas imputadas.
    """
    
    import numpy as np

    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3-q1
    upper = df[~(df>(q3+1.5*IQR))].max()
    lower = df[~(df<(q1-1.5*IQR))].min()
    df = np.where(df>upper, df.median(), np.where(df<lower, df.median(), df))

    return df


"""
Sección: Feature Selection
"""
def forward_selection(X, y, significance_level=0.01):
    """
    Función: Se declara función del método forward para la selección de características.
    Input: Dataframe(X), Serie(y) que son el resultado del balanceo de variables de 
    salida y variable con el nivel de significancia(por lo gral va entre 0.01(1%) y 
    0.05(5%).
    Output: Serie con la columnas seleccionadas.
    """

    import pandas as pd
    import statsmodels.api as sm

    initial_features = X.columns.tolist()
    best_features = []
    
    while(len(initial_features) > 0):
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features)
        
        for new_column in remaining_features:
            model = sm.OLS(y, sm.add_constant(X[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        
        if(min_p_value < significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
            
    return best_features


"""
Sección: Métricas, cálculo e impresión
"""
def ver_metricas_clasificacion(y_test, y_pred, y_pred_prob):
    """ 
    Función: Muestra los valores de las métricas para los algoritmos de clasificación.
    Input: Serie con la columna target de prueba(y_test), Serie con la columna 
    predecida(y_pred), Serie con la columna predecida probable(y_pred_prob).
    Output: Impresión de las métricas.
    """

    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, precision_score, recall_score


    acc = accuracy_score(y_test, y_pred, normalize=True)
    num_acc = accuracy_score(y_test, y_pred, normalize=False)

    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
        
    print("Cantidad de datos de test: ",len(y_test))
    #print("accuracy_count : " , num_acc)
    print("Accuracy: " , acc)
    print("Precision : " , prec)
    print("Recall : ", recall)
    print('ROC AUC Score:', roc_auc_score(y_test, y_pred_prob))
    print()


"""
Sección: Herramientas EDA
"""
def generar_profile(df, titulo='Resumen'):
    """ 
    Función: Muestra los valores de las métricas para los algoritmos de clasificación.
    Input: Dataframe y titulo pare el reporte.
    Output: Impresión del profile.
    """

    from pandas_profiling import ProfileReport

    profile = ProfileReport(df, title=titulo)
    profile.to_notebook_iframe()
    #profile.to_widgets()


"""
Sección: Encoding de variables 
"""
def codificar_ohe(df, lista_cols):
    """ 
    Función: Codifica con One Hot Enecoder las columas pasadas en la lista en el 
    dataframe pasado como otro parámetro.
    Input: Dataframe, Lista de columnas a codificar.
    Output: Dataframe con las columnas codificadas.
    """
    
    from sklearn.preprocessing import LabelEncoder
    
    ohe = LabelEncoder()

    for col in lista_cols:
        df[col] = ohe.fit_transform(df[col]).astype('int32')
    
    return df