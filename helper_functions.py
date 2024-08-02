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

def filter_dict(dict, keys):
    return {k:dict[k] for k in keys if k in dict.keys()}



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

def plot_feature_array(array, file_name, feature_names, feature_colors: dict):
    f = np.array(array).T
    if feature_names is None or len(feature_names) != len(f):
        feature_names = [f'feature {i+1}' for i in range(len(f))]

    plt.figure(figsize=(18.5, 6))
    ax = plt.subplot()
    for i, y in enumerate(f):
        plt.plot(np.arange(len(y)), y, label=feature_names[i], color=feature_colors[i])
    ax.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.01))
    plt.savefig(f"plots/{file_name}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def get_proportion(x):
        if np.sum(np.abs(x)) == 0: 
            return x
        return [round(val*100) for val in np.abs(x)/np.sum(np.abs(x))]

def scale_results(results: pd.DataFrame):
    for key, res_array in results.items():
        results[key] = get_proportion(res_array)
    return results

def plot_results(result_df: pd.DataFrame, MADs: dict, file_name, feature_colors: dict, padding=2, text_rotation=30, n_cols=4):
    plt.style.use('tableau-colorblind10')
    n_features = len(result_df.index)
    
    multiplier = -(n_features-1)/2
    width = 1/(n_features+padding)

    scale_results(result_df)

    x = np.arange(len(result_df.columns))
    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(18.5, 7.5)
    
    # plot result labels, grid, axis
    ax.set_xticks(x, [s.replace(' ', '\n') for s in result_df.columns], rotation=text_rotation)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(tick.PercentFormatter(100, decimals=0))
    ax.yaxis.grid(color='gray', linestyle='dashed')

    # plot bars
    for feature, results in result_df.iterrows():
        offset = width * multiplier
        ax.bar(x + offset, results, width, label=feature, color=feature_colors[feature])
        multiplier += 1

    # plot vertical lines between result categories
    for i in range(len(result_df.columns)-1):
        plt.axvline(x=i+0.5, color='black', lw = 0.5)

    # plot MADs with a separate axis
    ax_mad = ax.twinx()
    ax_mad.scatter(range(len(MADs)), list(MADs.values()), marker='x', color='black')

    ax.legend(loc='lower center', ncol=n_cols, bbox_to_anchor=(0.5, 1.01))

    # save as pdf and show plot
    plt.savefig(f"plots/{file_name}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def plot_all_results(results: list[pd.DataFrame], file_name, feature_colors: dict, padding=2, text_rotation=30, n_cols=4):
    plt.style.use('tableau-colorblind10')
    feature_keys = results[0].index
    result_keys = results[0].columns
    n_features = len(feature_keys)

    results_scaled = [scale_results(res) for res in results]

    mean_df = pd.DataFrame(index=feature_keys)
    min_df = pd.DataFrame(index=feature_keys)
    max_df = pd.DataFrame(index=feature_keys)
    for key in result_keys:
        res_arrays = np.array([result[key] for result in results_scaled]).T
        mean_df[key] = np.array([np.mean(array) for array in res_arrays])
        min_df[key] = mean_df[key]-np.array([np.min(array) for array in res_arrays])
        max_df[key] = np.array([np.max(array) for array in res_arrays])-mean_df[key]

    multiplier = -(n_features-1)/2
    width = 1/(n_features+padding)

    x = np.arange(len(result_keys))
    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(18.5, 7.3)
    ax.set_xticks(x, [s.replace(' ', '\n') for s in result_keys], rotation=text_rotation)

    for feature, mean in mean_df.iterrows():
        offset = width * multiplier
        error = [min_df.loc[feature], max_df.loc[feature]]
        ax.bar(x + offset, mean, width, label=feature, yerr=error, capsize=width*35, color=feature_colors[feature])
        multiplier += 1

    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(tick.PercentFormatter(100, decimals=0))
    ax.yaxis.grid(color='gray', linestyle='dashed')

    for i in range(len(result_keys)-1):
        plt.axvline(x=i+0.5, color='black', lw = 0.5)

    ax.legend(loc='lower center', ncol=n_cols, bbox_to_anchor=(0.5, 1))
    plt.savefig(f"plots/{file_name}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def plot_result(result_df: pd.DataFrame, MADs: list, file_name, title, feature_colors: dict, padding=2, text_rotation=0, n_cols=4):
    n_features = len(result_df.index)
    
    multiplier = -(n_features-1)/2
    width = 1/(n_features+padding)

    scale_results(result_df)

    x = np.arange(len(result_df.columns) + 1)
    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(18, 8)
    
    # plot result labels, grid, axis
    ax.set_xticks(x, [s.replace(' ', '\n') for s in result_df.columns] + ['all'], rotation=text_rotation)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(tick.PercentFormatter(100, decimals=0))
    ax.yaxis.grid(color='gray', linestyle='dashed')

    # plot bars
    for feature, results in result_df.iterrows():
        mean = np.mean(results)
        error = [[mean-np.min(results)], [np.max(results)-mean]]
        offset = width * multiplier
        ax.bar(x[:-1] + offset, results, width, label=feature, color=feature_colors[feature])
        ax.bar(x[-1]+ offset, mean, width, yerr=error, capsize=width*35, color=feature_colors[feature])
        multiplier += 1

    # plot vertical lines between result categories
    for i in range(len(x)-1):
        plt.axvline(x=i+0.5, color='black', lw = 0.5)

    # plot MADs with a separate axis
    ax_mad = ax.twinx()
    ax_mad.scatter(range(len(MADs)), MADs, marker='x', color='black')

    ax.set_title(title, y=1.22 + 0.13*((n_features-1) // n_cols))
    ax.legend(loc='lower center', ncol=n_cols, bbox_to_anchor=(0.5, 1))

    # save as pdf and show plot
    plt.savefig(f"plots/{file_name}.pdf", format="pdf", bbox_inches="tight")
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