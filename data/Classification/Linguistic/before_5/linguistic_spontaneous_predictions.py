#!/usr/bin/env python
# coding: utf-8
import sys
import speechbrain
from Classification.PCA_PLDA_EER_Classifier import PCA_PLDA_EER_Classifier
from statistics import mode
import random
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os
import sys
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

random_state = 20
random_seed = 20

test_only = 0

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def get_n_folds(arrayOfSpeaker):
    data = list(arrayOfSpeaker)  # list(range(len(arrayOfSpeaker)))
    num_of_folds = 10
    n_folds = []
    for i in range(num_of_folds):
        n_folds.append(data[int(i * len(data) / num_of_folds):int((i + 1) * len(data) / num_of_folds)])
    return n_folds

def normalize(train_split, test_split):

    train_set = train_split
    test_set = test_split

    feat_train = train_set[:, :-1]
    lab_train = train_set[:, -1:]
    lab_train = lab_train.astype('int')

    feat_test = test_set[:, :-1]
    lab_test = test_set[:, -1:]
    lab_test = lab_test.astype('int')

    X_train, X_test, y_train, y_test = feat_train, feat_test, lab_train, lab_test
    y_test = y_test.ravel()
    y_train = y_train.ravel()
    X_train = X_train.astype('float')
    X_test = X_test.astype('float')
    normalized_test_X = (X_test - X_train.mean(0)) / (X_train.std(0) + 0.01)
    normalized_train_X = (X_train - X_train.mean(0)) / (X_train.std(0) + 0.01)

    return normalized_train_X, normalized_test_X, y_train, y_test

all_metrics = []
all_predictions = pd.DataFrame(columns=['Embedding', 'Speaker', 'True_Label', 'Predicted_Label', 'Fold'])

embedding_val_f1s = {}  

feats_names = [
    'bert', 'cross-en-fr-roberta', 'distiluse', 'distiluse-v1',
    'e5-large','LaBSE','text2vec','xlm-roberta','lama3'
]

for feat_name in feats_names:
    print(f"Experiments with {feat_name}")
    feat_pth_pd = f'/five_years/spontaneous/AD/{feat_name}/'
    feat_pth_cn = f'/five_years/spontaneous/CTL/{feat_name}/'
    out_path = ''
    print(f"The output directory exists--> {os.path.isdir(out_path)}")

    path_files_pd = [os.path.join(feat_pth_pd, elem) for elem in sorted(os.listdir(feat_pth_pd)) if "concatenated" not in elem]
    names_pd = [os.path.basename(elem).split("_ID_")[0] for elem in path_files_pd]
    # add labels --> 0 for AD
    labels_pd = [0]*len(path_files_pd)
    df_pd = pd.DataFrame(list(zip(names_pd, path_files_pd, labels_pd)),
                   columns =['names', 'path_feat', 'labels'])

    path_files_cn = [os.path.join(feat_pth_cn, elem) for elem in sorted(os.listdir(feat_pth_cn)) if "concatenated" not in elem]
    names_cn = [os.path.basename(elem).split("_ID_")[0] for elem in path_files_cn]
    # add labels --> 1 for CN
    labels_cn = [1]*len(path_files_cn)
    df_cn = pd.DataFrame(list(zip(names_cn, path_files_cn, labels_cn)), columns = ['names', 'path_feat', 'labels'])

    df_stats = pd.read_excel('speakes_pairs.xlsx')
    cn_names = [elem.replace(" ", "_") for elem in df_stats['Name'].tolist()]
    pd_names = [elem.split("\t")[0].replace(" ", "_") for elem in df_stats['Use as a control for'].tolist()]
    list_names_pd = list(set(df_pd['names'].tolist()))
    list_names_cn = list(set(df_cn['names'].tolist()))
    pairs_of_names = list(zip(cn_names, pd_names))

    spain = pd.DataFrame()
    for paired in pairs_of_names:
        if paired[0] in list_names_cn and paired[1] in list_names_pd:
            gr_cn = df_cn.groupby("names").get_group(paired[0])
            gr_pd = df_pd.groupby("names").get_group(paired[1])
            if len(gr_pd) > len(gr_cn):
                gr_pd = gr_pd.sample(n=len(gr_cn), random_state=random_state)
            elif len(gr_cn) > len(gr_pd):
                gr_cn = gr_cn.sample(n=len(gr_pd), random_state=random_state)

            spain = pd.concat([spain, gr_cn], ignore_index=True)
            spain = pd.concat([spain, gr_pd], ignore_index=True)

    arrayOfSpeaker_cn = sorted(list(set(spain.groupby('labels').get_group(1)['names'].tolist())))
    random.Random(random_seed).shuffle(arrayOfSpeaker_cn)

    arrayOfSpeaker_pd =  sorted(list(set(spain.groupby('labels').get_group(0)['names'].tolist())))
    random.Random(random_seed).shuffle(arrayOfSpeaker_pd)

    cn_sps = get_n_folds(arrayOfSpeaker_cn)
    pd_sps = get_n_folds(arrayOfSpeaker_pd)

    data = []
    for cn_sp, pd_sp in zip(sorted(cn_sps, key=len), sorted(pd_sps, key=len, reverse=True)):
        data.append(cn_sp + pd_sp)
    n_folds = sorted(data, key=len, reverse=True)

    folds = []
    for fold in n_folds:
        data_fold = []
        speaker_names_fold = []
        data_i = spain[spain["names"].isin(fold)]
        names = list(set(data_i['names'].tolist()))
        for name in names:
            gr_sp = data_i.groupby('names').get_group(name)
            label_row = (gr_sp['labels'].tolist())[0]
            read_vect = [np.load(vec) for vec in gr_sp['path_feat'].tolist()]
            read_vect = [vec / np.linalg.norm(vec) for vec in read_vect]
            feat = np.mean(read_vect, axis=0)
            feat = feat / np.linalg.norm(feat)
            feat = np.append(feat, label_row)
            data_fold.append(feat)
            speaker_names_fold.append(name)
        folds.append((np.array(data_fold), speaker_names_fold))

    # Data for training and testing in each fold
    data_train_test = [
        (np.concatenate([folds[j][0] for j in range(10) if j != i]),  # train
         folds[i][0],                                                # test
         folds[i][1])                                                # test speakers
        for i in range(10)
    ]

    # ---------------------------
    # Determine the optimal PCA_n parameter and record the average F1 score from inner loop
    # ---------------------------
    if test_only == 0:
        best_params = []
        fold_val_scores = []
        for i, (data_train, data_test, speaker_names_test) in enumerate(data_train_test, 1):
            print(i)
            normalized_train_X, normalized_test_X, y_train, y_test = normalize(data_train, data_test)
            tuned_params = {"PCA_n": [3,5,7,10,15,20,25]}
            model = PCA_PLDA_EER_Classifier(normalize=0)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=tuned_params,
                n_jobs=-1,
                cv=cv,
                scoring='f1',
                error_score=0
            )
            grid_result = grid_search.fit(normalized_train_X, y_train)
            print(grid_result.best_params_)
            best_params.append(grid_result.best_params_['PCA_n'])

            # Record the best F1 score from inner loop
            fold_val_scores.append(grid_result.best_score_)

        # Get the mode of optimal PCA_n
        best_param = mode(best_params)
        print('**********best pca n:')
        print(best_param)

        # Store the average best F1 score for this embedding on the validation set
        avg_val_f1 = np.mean(fold_val_scores)
        embedding_val_f1s[feat_name] = avg_val_f1
    else:
        # For testing only, assume a predefined best_param value
        best_param = 10

    # ---------------------------
    # Training and Testing
    # ---------------------------
    predictions_df = pd.DataFrame(columns=['Embedding', 'Speaker', 'True_Label', 'Predicted_Label', 'Fold'])

    thresholds = []
    predictions = []
    truth = []
    test_scores = []
    for i, (data_train, data_test, speaker_names_test) in enumerate(data_train_test, 1):
        print(i)
        normalized_train_X, normalized_test_X, y_train, y_test = normalize(data_train, data_test)
        y_test = y_test.tolist()
        model = PCA_PLDA_EER_Classifier(PCA_n=best_param, normalize=0)
        model.fit(normalized_train_X, y_train)
        grid_predictions = model.predict(normalized_test_X)
        print(model.eer_threshold)
        grid_test_scores = model.predict_scores_list(normalized_test_X)

        predictions += grid_predictions
        truth += y_test
        test_scores += grid_test_scores[:, 0].tolist()
        thresholds += [model.eer_threshold]*len(y_test)

        fold_speakers = speaker_names_test
        if len(fold_speakers) > len(y_test):
            fold_speakers = fold_speakers[:len(y_test)]
        elif len(fold_speakers) < len(y_test):
            y_test = y_test[:len(fold_speakers)]
            grid_predictions = grid_predictions[:len(fold_speakers)]

        fold_predictions_df = pd.DataFrame({
            'Embedding': [feat_name] * len(fold_speakers),
            'Speaker': fold_speakers,
            'True_Label': y_test,
            'Predicted_Label': grid_predictions,
            'Fold': [i] * len(fold_speakers)
        })
        predictions_df = pd.concat([predictions_df, fold_predictions_df], ignore_index=True)

    # Record the test set report for this embedding
    all_predictions = pd.concat([all_predictions, predictions_df], ignore_index=True)

    print()
    print('----------')
    print('----------')
    print("Final results (Test)")
    print(classification_report(truth, predictions, output_dict=False))
    cm = confusion_matrix(truth, predictions)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    specificity = round(tn / (tn + fp), 2)
    sensitivity = round(tp / (tp + fn), 2)
    accuracy = round((tp + tn) / (tp + tn + fp + fn), 2)
    auroc = round(roc_auc_score(truth, test_scores), 2)
    f1 = round(classification_report(truth, predictions, output_dict=True)['weighted avg']['f1-score'], 2)
    print('specificity:', specificity)
    print('sensitivity:', sensitivity)
    print('ROC_AUC:', auroc)
    print('*************')
    print('*************')

    # Save the final metrics
    metrics_df = pd.DataFrame({
        'feat': [feat_name],
        'AUROC': [auroc],
        'F1 Score': [f1],
        'Sensitivity': [sensitivity],
        'Specificity': [specificity],
        'Accuracy': [accuracy],
        'Best PCA Param': [best_param]
    })

    file_out = os.path.join(out_path, feat_name + "_PCA_results.csv")
    metrics_df.to_csv(file_out, index=False)
    all_metrics.append(metrics_df)

# Aggregate the test results for all embeddings
final_metrics_df = pd.concat(all_metrics, ignore_index=True)
file_out = os.path.join(out_path, "PCA_results.csv")
final_metrics_df.to_csv(file_out, index=False)

# Save detailed predictions on Test set
all_predictions_file_out = os.path.join(out_path, "all_embeddings_predictions.csv")
all_predictions.to_csv(all_predictions_file_out, index=False)

# ============ Output the top three embeddings with the highest validation F1 scores and save them as a CSV file ============
if embedding_val_f1s:
    val_scores_df = pd.DataFrame({
        'Embedding': list(embedding_val_f1s.keys()),
        'MeanValF1': list(embedding_val_f1s.values())
    })
    val_scores_df = val_scores_df.sort_values(by='MeanValF1', ascending=False)
    # Top3
    top3 = val_scores_df.head(4)
    top3_csv_out = os.path.join(out_path, "top3_validation_f1.csv")
    top3.to_csv(top3_csv_out, index=False)

    print("\n======================================")
    print("Top 3 Embeddings from inner loops:")
    print(top3)
    print(f"Results saved to: {top3_csv_out}")
