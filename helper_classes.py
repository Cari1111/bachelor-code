import pypsa
import pickle 
import pandas as pd
import os
from typing import Callable
import warnings

from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from helper_functions import get_result_dict, plot_feature_array, plot_results, plot_all_results, plot_result, MAD, filter_dict

class ResultVectors(dict):
    def __setitem__(self, key, file_name):
        with open(f'{file_name}.pkl', 'rb') as f:
            result_dict = pickle.load(f)
        super().__setitem__(key, result_dict)

class NetworkSplits(dict):
    def __setitem__(self, key, directory_name):
        directory = os.fsencode(directory_name)
        n_files = len([f for f in os.listdir(directory)])
        netw_split = [pypsa.Network(f'{directory_name}/{i}.nc') for i in range(n_files)]
        super().__setitem__(key, netw_split)

class Experiment():
    def __init__(self, name:str, netw_splits: NetworkSplits, result_vectors=None, feature_matrices=None, feature_names=None, feature_colors=None) -> None:
        self.name = name
        self.split_names = list(netw_splits.keys())
        self.netw_splits = netw_splits
        if result_vectors is None or self.split_names != list(result_vectors.keys()):
            result_vectors = dict()
            for key, netw_split in netw_splits.items():
                result_vectors[key] = get_result_dict(netw_split)
            result_vectors = ResultVectors(result_vectors)
        self.result_vectors=result_vectors
        self.feature_matrices = feature_matrices
        self.feature_names = feature_names
        self.feature_colors = feature_colors
        self.results = dict()

    def get_n_cols_feature(self):
        n_features = len(self.feature_names)
        if n_features <= 4: return n_features
        for n in [4,3]:
            if n_features % n == 0: return n
        return 4
    
    def create_features(self, extract_feature_func: Callable, feature_names:list[str]=None, feature_colors=None):
        if self.feature_matrices is None:
            self.feature_matrices = dict()
            
        for split in self.split_names:
            matrix = []
            for i, netw in enumerate(self.netw_splits[split]):
                matrix.append(extract_feature_func(netw, i, split))
            matrix = StandardScaler().fit_transform(matrix)
            self.feature_matrices[split] = matrix
        self.feature_names = feature_names
        self.feature_colors = feature_colors
        if self.feature_names is None:
            self.feature_names = [f'feature {i+1}' for i in range(len(matrix[0]))]

    
    def run_algorithms(self):
        for split in self.split_names:
            coefficients = pd.DataFrame(index=self.feature_names)
            weights = pd.DataFrame(index=self.feature_names)
            MADs = dict()

            for result_key, result_array in self.result_vectors[split].items():
                lasso_regr = LassoCV()
                RF_regr = RandomForestRegressor(max_features="sqrt", max_samples=0.7)
                with warnings.catch_warnings(action="ignore"):
                    lasso_regr.fit(self.feature_matrices[split], result_array)
                    RF_regr.fit(self.feature_matrices[split], result_array)
                
                coefficients[result_key] = lasso_regr.coef_
                weights[result_key] = RF_regr.feature_importances_
                MADs[result_key] = MAD(result_array) 
                
            self.results[split] = (coefficients, weights, MADs)
    
    def get_color_dict(self):
        return {feature: self.feature_colors[i] for i, feature in enumerate(self.feature_names)}

    def plot_features(self, split:str):
        plot_feature_array(
            self.feature_matrices[split], 
            file_name=f'{self.name}_{split}_features',
            feature_names=self.feature_names,
            feature_colors=self.feature_colors
        )
    
    def plot_results(self, split:str, result_categories:list[str]|None=None, text_rotation=30, n_cols=None):
        category_names = ''
        coefficients, weights, MADs = self.results[split]
        if result_categories is not None:
            coefficients = coefficients.filter(result_categories)
            weights = weights.filter(result_categories)
            MADs = filter_dict(MADs, result_categories)
            category_names = '_' + ''.join([n[0] for n in result_categories])
        else:
            coefficients, weights, MADs = self.results[split]
        for result_df, algo in [(coefficients, 'Lasso'), (weights, 'Random_Forests')]:
            plot_results(
                result_df, 
                MADs=MADs, 
                file_name=f'{self.name}_{split}{category_names}_{algo}', 
                feature_colors=self.get_color_dict(), 
                text_rotation=text_rotation, 
                n_cols=n_cols or self.get_n_cols_feature())
        

    def plot_all_results(self, result_categories:list[str]|None=None, text_rotation=30, n_cols=None):
        if result_categories is None:
            category_names = ''
            result_categories = list(self.result_vectors[self.split_names[0]].keys())
            
        else:
            category_names = '_' + ''.join([n[0] for n in result_categories])
        all_results = []
        for coefficients, weights, _ in self.results.values():
            all_results.append(coefficients.filter(result_categories))
            all_results.append(weights.filter(result_categories)) 
        plot_all_results(all_results, file_name=f'{self.name}{category_names}_all_results', feature_colors=self.get_color_dict(), text_rotation=text_rotation, n_cols=n_cols or self.get_n_cols_feature())

    def plot_one_result(self, result_category, text_rotation=0):
        df =pd.DataFrame(index=self.feature_names)
        MADs= []
        for split in self.split_names:
            coefficients, weights, mad = self.results[split]
            df[f'Lasso {split}'] = coefficients[result_category]
            df[f'Random Forests {split}'] = weights[result_category]
            MADs.append(mad[result_category])
            MADs.append(mad[result_category])
        plot_result(df, MADs, f'{self.name}_{result_category}', result_category, self.get_color_dict(), text_rotation=text_rotation, n_cols=self.get_n_cols_feature())