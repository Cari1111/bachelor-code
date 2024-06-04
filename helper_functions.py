import pypsa
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as tick
import pandas as pd

from scipy import stats
from sklearn.preprocessing import StandardScaler

def split_list(list, n):
    k, m = divmod(len(list), n)
    return [list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]



def plot_generator(netw_splits, name: str):
    for netw in netw_splits:
        netw.generators_t.p_max_pu[name].plot(figsize=(16,5))

def plot_feature_array(array):
    f = np.array(array).T
    for i, y in enumerate(f):
        plt.plot(np.arange(len(y)), y, label=f'feature {i+1}')
    plt.legend()
    plt.show()

def plot_results(coef_dict: dict, feature_names, padding=2):
    plt.style.use('tableau-colorblind10')
    n_features = len(list(coef_dict.values())[0])
    
    multiplier = -(n_features-1)/2
    width = 1/(n_features+padding)

    coefs_scaled = np.array([np.abs(x)/np.sum(np.abs(x)) for x in list(coef_dict.values())]).T
    coefs = [[round(y*100) for y in x] for x in coefs_scaled]

    x = np.arange(len(coef_dict))
    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(18.5, 6)
    ax.set_xticks(x, [s.replace(' ', '\n') for s in coef_dict.keys()])

    for i, coef in enumerate(coefs):
        offset = width * multiplier
        rects = ax.bar(x + offset, tuple(coef), width, label=feature_names[i])
        ax.bar_label(rects, padding=3, rotation=90, fmt='%.0f%%')
        multiplier += 1

    ax.yaxis.set_major_formatter(tick.PercentFormatter(100))
    ax.margins(y=0.15)

    for i in range(len(coef_dict)-1):
        plt.axvline(x=i+0.5, color='black', lw = 0.5)

    ax.legend(loc='lower center', ncols=10, bbox_to_anchor=(0.5, 1.01))




def get_result_array(netw_splits: list[pypsa.Network], key):
    result_array = []
    for netw in netw_splits:
        result = np.mean(netw.statistics()[key])
        result_array.append(result)
    return stats.zscore(result_array)

def get_feature(df: pd.DataFrame, search: str):
    filtered_df = df.filter(like=search, axis=1)
    return np.mean(np.array(filtered_df), axis=(0,1))

def get_feature_array(netw_splits: list[pypsa.Network], feature_key_words):
    feature_array = []
    for netw in netw_splits:
        feature_data = netw.generators_t.p_max_pu
        features = [get_feature(feature_data, key_word) for key_word in feature_key_words]
        feature_array.append(features)
    
    feature_scaler = StandardScaler().fit(feature_array)
    feature_array_norm = feature_scaler.transform(feature_array)

    return feature_array_norm