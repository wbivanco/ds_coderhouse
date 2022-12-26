import numpy as np
import pandas as pd
import statsmodels.api as sm


def forward_selection(data, target, significance_level=0.01):
    """
    Se elige y se declara la función del método forward de selección de características.
    """
    initial_features = data.columns.tolist()
    best_features = []
    
    while (len(initial_features) > 0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        
        if(min_p_value < significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
            
    return best_features


def imputar_outliers_IQR(df):
    """
    Se crea una función que permite aplicar el criterio de reemplazo de atípicos. Se reemplaza el
    atípico por la mediana que es una medida robusta(no influenciada por outliers). 
    RecibeSe un dataframe con las columnas a imputar.
    """
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3-q1
    upper = df[~(df>(q3+1.5*IQR))].max()
    lower = df[~(df<(q1-1.5*IQR))].min()
    df = np.where(df > upper, 
                 df.median(), 
                 np.where(df < lower, 
                          df.median(), 
                          df) 
                 )
    return df


def formatear_autopct(values):
    """
    Permite agregar al piechart, info de la cantida de cada porción, no solo los porcentajes.
    """
    def my_format(pct):
        total = sum(values)        
        val = int(round(pct*total/100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format
