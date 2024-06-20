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

def MAD(x):
    return np.median(np.absolute(x - np.mean(x)))



def plot_generator(netw_splits, generator: str):
    plt.style.use('tableau-colorblind10')
    means_val = []
    means_dates = []
    for netw in netw_splits:
        netw.generators_t.p_max_pu[generator].plot(figsize=(16,5))
        means_val.append(np.mean(netw.generators_t.p_max_pu[generator]))
        means_dates.append(netw.snapshots[int(len(netw.snapshots)/2)])
    plt.plot(means_dates, means_val, color='black')
    plt.savefig(f"plot/generator_{generator}.pdf", format="pdf", bbox_inches="tight")

def plot_feature_array(array, feature_names = None):
    f = np.array(array).T
    if feature_names is None or len(feature_names) != len(f):
        feature_names = [f'feature {i+1}' for i in range(len(f))]

    plt.figure(figsize=(16,5))
    for i, y in enumerate(f):
        plt.plot(np.arange(len(y)), y, label=feature_names[i])
    plt.legend()
    plt.show()

def plot_results(result_dict: dict, feature_names, MADs, title, padding=2):
    plt.style.use('tableau-colorblind10')
    n_features = len(list(result_dict.values())[0])
    
    multiplier = -(n_features-1)/2
    width = 1/(n_features+padding)

    def get_proportion(x):
        if np.sum(np.abs(x)) == 0: 
            return x
        return np.abs(x)/np.sum(np.abs(x))

    coefs_scaled = np.array([get_proportion(x) for x in list(result_dict.values())]).T
    coefs = [[round(y*100) for y in x] for x in coefs_scaled]

    x = np.arange(len(result_dict))
    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(18.5, 6)
    ax.set_xticks(x, [s.replace(' ', '\n') for s in result_dict.keys()])

    for i, coef in enumerate(coefs):
        offset = width * multiplier
        rects = ax.bar(x + offset, tuple(coef), width, label=feature_names[i])
        ax.bar_label(rects, padding=3, rotation=90, fmt='%.0f%%')
        multiplier += 1

    ax.yaxis.set_major_formatter(tick.PercentFormatter(100))
    ax.margins(y=0.15)

    for i in range(len(result_dict)-1):
        plt.axvline(x=i+0.5, color='black', lw = 0.5)

    ax_mad = ax.twinx()
    ax_mad.scatter(range(len(MADs)), MADs, marker='x', color='black')

    ax.legend(loc='lower center', ncols=10, bbox_to_anchor=(0.5, 1.01))
    plt.savefig(f"plots/{title}.pdf", format="pdf", bbox_inches="tight")
    plt.show()




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