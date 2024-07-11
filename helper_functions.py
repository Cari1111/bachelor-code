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
    plt.savefig(f"plots/generator_{generator}.pdf", format="pdf", bbox_inches="tight")

def plot_feature_array(array, file_name, feature_names = None):
    f = np.array(array).T
    if feature_names is None or len(feature_names) != len(f):
        feature_names = [f'feature {i+1}' for i in range(len(f))]

    plt.figure(figsize=(18.5, 6))
    ax = plt.subplot()
    for i, y in enumerate(f):
        plt.plot(np.arange(len(y)), y, label=feature_names[i])
    ax.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.5, 1.01))
    plt.savefig(f"plots/{file_name}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def get_proportion(x):
        if np.sum(np.abs(x)) == 0: 
            return x
        return [round(val*100) for val in np.abs(x)/np.sum(np.abs(x))]

def scale_results(results):
    for key, res_array in results.items():
        results[key] = get_proportion(res_array)
    return results

def convert_to_feature_dict(result_dict, feature_names):
    feature_dict = dict()
    for i, key in enumerate(feature_names):
        feature_dict[key] = [array[i] for _, array in result_dict.items()]
    return feature_dict

def plot_results(result_dict: dict, feature_names, MADs, title, padding=2):
    plt.style.use('tableau-colorblind10')
    n_features = len(list(result_dict.values())[0])
    if (n_features != len(feature_names)):
        raise Exception('number of feature names does not match with features')
    
    multiplier = -(n_features-1)/2
    width = 1/(n_features+padding)

    scale_results(result_dict)
    feature_dict = convert_to_feature_dict(result_dict, feature_names)

    x = np.arange(len(result_dict))
    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(18.5, 6)
    
    # plot result labels, grid, axis
    ax.set_xticks(x, [s.replace(' ', '\n') for s in result_dict.keys()], rotation=25)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(tick.PercentFormatter(100))
    ax.yaxis.grid(color='gray', linestyle='dashed')

    # plot bars
    for feature, results in feature_dict.items():
        offset = width * multiplier
        ax.bar(x + offset, results, width, label=feature)
        multiplier += 1

    # plot vertical lines between result categories
    for i in range(len(result_dict)-1):
        plt.axvline(x=i+0.5, color='black', lw = 0.5)

    # plot MADs with a separate axis
    ax_mad = ax.twinx()
    ax_mad.scatter(range(len(MADs)), MADs, marker='x', color='black')

    ax.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.5, 1.01))
    plt.savefig(f"plots/{title}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def plot_all_results(results: list[dict], feature_names, title, padding=2):
    plt.style.use('tableau-colorblind10')
    n_features = len(list(results[0].values())[0])
    common_keys = results[0].keys()

    results_scaled = [scale_results(res) for res in results]

    mean_dict = dict()
    min_dict = dict()
    max_dict = dict()
    for key in common_keys:
        res_arrays = np.array([result[key] for result in results_scaled]).T
        mean_dict[key] = np.array([np.mean(array) for array in res_arrays])
        min_dict[key] = mean_dict[key]-np.array([np.min(array) for array in res_arrays])
        max_dict[key] = np.array([np.max(array) for array in res_arrays])-mean_dict[key]

    feature_dict = dict()
    for i, key in enumerate(feature_names):
        means = [array[i] for _, array in mean_dict.items()]
        min = [array[i] for _, array in min_dict.items()]
        max = [array[i] for _, array in max_dict.items()]
        feature_dict[key] = (means, min, max)

    multiplier = -(n_features-1)/2
    width = 1/(n_features+padding)

    x = np.arange(len(common_keys))
    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(18.5, 6)
    ax.set_xticks(x, [s.replace(' ', '\n') for s in common_keys], rotation=25)

    for feature, (mean, min, max) in feature_dict.items():
        offset = width * multiplier
        ax.bar(x + offset, mean, width, label=feature, yerr=[min, max], capsize=int(6))
        multiplier += 1

    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(tick.PercentFormatter(100))
    ax.yaxis.grid(color='gray', linestyle='dashed')

    for i in range(len(results[0])-1):
        plt.axvline(x=i+0.5, color='black', lw = 0.5)

    ax.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.5, 1.01))
    plt.savefig(f"plots/{title}.pdf", format="pdf", bbox_inches="tight")
    plt.show()






def get_result_array(netw_splits: list[pypsa.Network], key):
    result_array = []
    for netw in netw_splits:
        result = np.mean(netw.statistics()[key])
        result_array.append(result)
    return stats.zscore(result_array)

def get_result_dict(netw_split):
    result_dict = dict()
    result_keys = [column for column in netw_split[0].statistics()]
    for result_key in result_keys:
        result_array = get_result_array(netw_split, result_key)
        if any(np.isinf(res) or np.isnan(res) for res in result_array):
            continue
        result_dict[result_key] = result_array
    return result_dict

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