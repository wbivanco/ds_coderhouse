
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
    Función: Grafica boxplot con o sin atípicos del dataframe según el valor del parámetro enviado.
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


def formatear_autopct(data):
    """
    Función: Permite agregar al piechart, info de la cantidad en número de cada porción del piechart, no solo los 
    porcentajes. Para que no dé error la función se debe llamar my_format y debe recibir pct como argumento
    por defecto.
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

    return mean(absolute(data - mean(data, axis)), axis)


def calcular_porcentajes_nulos(df):
    """
    Función: Calcula el porcentaje de nulos de un df.
    Input: Dataframe
    Ouput: Serie con los porcentajes de nulos.
    """

    porcen_nulos = (df.isnull().sum()/df.shape[0])*100
    porcen_nulos = porcen_nulos.sort_values(ascending=False)
    porcen_nulos = porcen_nulos[porcen_nulos>0]

    return porcen_nulos


"""
Sección: Imputación
"""
def imputar_outliers_IQR(df):
    """
    Función: permite aplicar el criterio de reemplazo de atípicos. Se reemplaza el
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
    Input: Dataframe(X), Serie(y) que son el resultado del balanceo de variables de salida 
    y variable con el nivel de significancia(por lo gral va entre 0.01(1%) y 0.05(5%).
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
    Función: muestra los valores de las métricas para los algoritmos de clasificación.
    Input: Serie con la columna target de prueba(y_test), Serie con la columna predecida(y_pred), Serie con la columna
    predecida probable(y_pred_prob).
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
    Función: muestra los valores de las métricas para los algoritmos de clasificación.
    Input: Dataframe y titulo pare el reporte.
    Output: Impresión del profile.
    """

    from pandas_profiling import ProfileReport

    profile = ProfileReport(df, title=titulo)
    profile.to_notebook_iframe()
    #profile.to_widgets()