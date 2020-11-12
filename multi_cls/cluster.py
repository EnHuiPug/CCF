import json
import copy
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn import metrics

def evaluate(y, y_):
    assert len(y) == len(y_)
    homogeneity_score = metrics.homogeneity_score(y, y_),
    completeness_score = metrics.completeness_score(y, y_),
    v_measure_score = metrics.v_measure_score(y, y_),
    adjusted_rand_score= metrics.adjusted_rand_score(y, y_),
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(y,  y_),
    return homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
        
def main():
    parser = argparse.ArgumentParser(description="Run feature cluster")
    parser.add_argument('--source_data', default='./data/unlabeled_data.csv')
    parser.add_argument('--features', default='')
    parser.add_argument('--finetune', default=False, type=bool)
    parser.add_argument('--task', default='labeled')
    args = parser.parse_args()
    print(args)
    
    def load_data(filepath):
        df = pd.read_csv(filepath)
        text = []
        for t in df['content']:
            text.append(t)
        return text
    
    if args.task == 'unlabeled':
        texts = load_data(args.source_data)
        features = np.loadtxt(args.features)
        print('Len feature data: ', len(features))
        assert len(texts) == len(features)

        clf = KMeans(n_clusters=10, random_state=10).fit(features)
        pred_labels = clf.labels_

        result = []
        for t, l in zip(texts, pred_labels):
            result.append([t,int(l)])
        df = pd.DataFrame(result, columns=['content', 'p_label'])
        df.to_csv('./output/unlabeled_data_cluster_first_round.csv', index=False)
        

    elif args.task == 'labeled':
        cls_features = np.loadtxt('./output/finetune_cls_features.txt')
        mean_features = np.loadtxt('./output/finetune_mean_features.txt')
        labels = np.loadtxt('./output/labels.txt').astype(int)

        cls_clf = KMeans(n_clusters=7, random_state=42).fit(cls_features)
        mean_clf = KMeans(n_clusters=7, random_state=42).fit(mean_features)

        print('cls_clf benchmark')
        homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score = evaluate(labels, cls_clf.labels_)
        print('homogeneity_score: {}, completeness_score: {}, v_measure_score: {}, adjusted_rand_score: {}, adjusted_mutual_info_score: {}'.format(
            homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
        ))

        print('mean_clf benchmark')
        homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score = evaluate(labels, mean_clf.labels_)
        print('homogeneity_score: {}, completeness_score: {}, v_measure_score: {}, adjusted_rand_score: {}, adjusted_mutual_info_score: {}'.format(
            homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
        ))

if __name__=='__main__':
    main()

# pretrain_model
# cls_clf benchmark
# homogeneity_score: (0.5978667645178977,), completeness_score: (0.6238762874643061,), v_measure_score: (0.6105946693792299,), adjusted_rand_score: (0.5283745390940591,), adjusted_mutual_info_score: (0.6100674478831313,)
# mean_clf benchmark
# homogeneity_score: (0.6244212702008646,), completeness_score: (0.6598051129462685,), v_measure_score: (0.6416257322191017,), adjusted_rand_score: (0.5340842724130763,), adjusted_mutual_info_score: (0.641137498211767,)

# finetune model
# cls_clf benchmark
# homogeneity_score: (0.9624474557057984,), completeness_score: (0.962456504044955,), v_measure_score: (0.96245197985411,), adjusted_rand_score: (0.9735714132150818,), adjusted_mutual_info_score: (0.9624022198035889,)
# mean_clf benchmark
# homogeneity_score: (0.9069249803502568,), completeness_score: (0.9074704202588714,), v_measure_score: (0.907197618320038,), adjusted_rand_score: (0.9193661459482131,), adjusted_mutual_info_score: (0.907074596239707,)