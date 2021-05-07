import re

def outlier_filter(arr):
    IQR=df[arr].quantile(0.75) - df[arr].quantile(0.25)
    lower_boundaries = df[arr].quantile(0.25) - (3 * IQR)
    upper_boundaries = df[arr].quantile(0.75) + (3 * IQR)
    df.drop(df[df[arr] > upper_boundaries].index,inplace=True)
    return [lower_boundaries,upper_boundaries,arr]
    
def one_off_filter(pred,act):
    i = 0
    for x,y in zip(pred,act):
        if x-y <=1 :
            pred[i]=y
        
        i = i + 1