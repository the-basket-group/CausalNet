import random
from os import path
import os
import shutil
import cv2
import time
import json
from copy import deepcopy

import pandas
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
from distutils.util import strtobool
import torch

import numpy as np


from Models import CausalNet


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):

            layer.reset_parameters()

def confusionMatrix(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall


def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'Negative': 0, 'Positive': 1, 'surprise': 2}
    # Display recognition result
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''



def whole_face_block_coordinates():
    df = pandas.read_csv('combined_3_class2_for_optical_flow.csv')
    m, n = df.shape

    face_block_coordinates = {}


    for i in range(0, m):
        image_name = str(df['sub'][i]) + '_' + str(
            df['filename_o'][i]) + ' .png'

        batch_landmarks=None

        if batch_landmarks is None:

            batch_landmarks = np.array([[[9.528073, 11.062551]
                                            , [21.396168, 10.919773]
                                            , [15.380184, 17.380562]
                                            , [10.255435, 22.121233]
                                            , [20.583706, 22.25584]]])

        row_n, col_n = np.shape(batch_landmarks[0])

        for i in range(0, row_n):
            for j in range(0, col_n):
                if batch_landmarks[0][i][j] < 7:
                    batch_landmarks[0][i][j] = 7
                if batch_landmarks[0][i][j] > 21:
                    batch_landmarks[0][i][j] = 21


        batch_landmarks = batch_landmarks.astype(int)

        face_block_coordinates[image_name] = batch_landmarks[0]
        tmp=image_name.split(' ')[0]
        tmp1=tmp+'_1 .png'
        face_block_coordinates[tmp1] = batch_landmarks[0]
        tmp2 = tmp + '_2 .png'
        face_block_coordinates[tmp2] = batch_landmarks[0]
        tmp3 = tmp + '_3 .png'
        face_block_coordinates[tmp3] = batch_landmarks[0]

    return face_block_coordinates


def crop_optical_flow_block():
    face_block_coordinates_dict = whole_face_block_coordinates()

    whole_optical_flow_path = './datasets/STSNet_whole_norm_u_v_os'
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)
    four_parts_optical_flow_imgs = {}

    for n_img in whole_optical_flow_imgs:
        four_parts_optical_flow_imgs[n_img]=[]
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)
        four_part_coordinates = face_block_coordinates_dict[n_img]
        l_eye = flow_image[four_part_coordinates[0][0]-7:four_part_coordinates[0][0]+7,
                four_part_coordinates[0][1]-7: four_part_coordinates[0][1]+7]
        l_lips = flow_image[four_part_coordinates[1][0] - 7:four_part_coordinates[1][0] + 7,
                four_part_coordinates[1][1] - 7: four_part_coordinates[1][1] + 7]
        nose = flow_image[four_part_coordinates[2][0] - 7:four_part_coordinates[2][0] + 7,
                four_part_coordinates[2][1] - 7: four_part_coordinates[2][1] + 7]
        r_eye = flow_image[four_part_coordinates[3][0] - 7:four_part_coordinates[3][0] + 7,
                four_part_coordinates[3][1] - 7: four_part_coordinates[3][1] + 7]
        r_lips = flow_image[four_part_coordinates[4][0] - 7:four_part_coordinates[4][0] + 7,
                four_part_coordinates[4][1] - 7: four_part_coordinates[4][1] + 7]
        four_parts_optical_flow_imgs[n_img].append(l_eye)
        four_parts_optical_flow_imgs[n_img].append(l_lips)
        four_parts_optical_flow_imgs[n_img].append(nose)
        four_parts_optical_flow_imgs[n_img].append(r_eye)
        four_parts_optical_flow_imgs[n_img].append(r_lips)

    return four_parts_optical_flow_imgs


def subject_of(name):
    # Subject id is the first underscore-token: '006_006_1_2'->'006', 'sub09_..'->'sub09', 's04_..'->'s04'.
    return name.split('_')[0]


def stack_four(name, flows):
    # Build the 4-frame [base, _1, _2, _3] input for one clip from its cropped 5-part flow.
    base = name.split(' ')[0]
    frames = []
    for key in (name, base + '_1 .png', base + '_2 .png', base + '_3 .png'):
        f = flows[key]
        l_eye_lips = cv2.hconcat([f[0], f[1]])
        r_eye_lips = cv2.hconcat([f[3], f[4]])
        frames.append(cv2.vconcat([l_eye_lips, r_eye_lips]))
    return frames


def read_split(main_path, subname, split, flows):
    # Read one split folder (u_train / u_test) -> (X 4-frame stacks, y labels, subject ids).
    X, y, subj = [], [], []
    root = os.path.join(main_path, subname, split)
    for cls in os.listdir(root):
        for n_img in os.listdir(os.path.join(root, cls)):
            X.append(stack_four(n_img, flows))
            y.append(int(cls))
            subj.append(subject_of(n_img))
    return X, y, subj


def make_loader(X, y, batch_size, shuffle=False):
    x = torch.Tensor(np.array(X)).permute(0, 1, 4, 2, 3)
    y = torch.Tensor(y).to(dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def main(config):
    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Each (attn_mode, seed) run writes its per-subject txts into its own folder so the
    # three variants never clobber each other. calculate_all_results.py --results_dir reads it.
    run_name = config.run_name or ('%s_seed%d' % (config.attn_mode, seed))
    results_dir = os.path.join('.', 'results', run_name)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    print('attn_mode=%s | seed=%d | results_dir=%s' % (config.attn_mode, seed, results_dir))

    learning_rate = 0.00005
    batch_size = 256 * 4
    epochs = config.epochs
    n_val = 4  # subjects held out of each fold's training pool for validation
    patience, min_delta = config.patience, 1e-4  # early stop on validation-UF1 plateau
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loss_fn = nn.CrossEntropyLoss()
    print('lr=%f, epochs=%d, patience=%d, n_val=%d, device=%s\n' % (learning_rate, epochs, patience, n_val, device))

    main_path = './datasets/three_norm_u_v_os'
    subName = os.listdir(main_path)
    all_five_parts_optical_flow = crop_optical_flow_block()
    print(subName)

    total_gt, total_pred = [], []
    histories = {}
    t = time.time()

    for n_subName in subName:
        print('Subject:', n_subName)
        # LOSO fold: this subject is the test set; all others form the training pool.
        X_pool, y_pool, subj_pool = read_split(main_path, n_subName, 'u_train', all_five_parts_optical_flow)
        X_test, y_test, _ = read_split(main_path, n_subName, 'u_test', all_five_parts_optical_flow)

        # Carve a subject-independent validation set out of the training pool (nested LOSO).
        train_subjects = sorted(set(subj_pool))
        rng = np.random.default_rng(seed)
        val_subjects = set(rng.choice(train_subjects, size=min(n_val, len(train_subjects) - 1), replace=False))
        tr = [i for i, s in enumerate(subj_pool) if s not in val_subjects]
        va = [i for i, s in enumerate(subj_pool) if s in val_subjects]

        train_dl = make_loader([X_pool[i] for i in tr], [y_pool[i] for i in tr], batch_size, shuffle=True)
        val_dl = make_loader([X_pool[i] for i in va], [y_pool[i] for i in va], batch_size)
        test_dl = make_loader(X_test, y_test, batch_size)

        model = CausalNet(
            image_size=28,
            patch_size=7,
            dim=256,
            heads=3,
            num_hierarchies=3,
            block_repeats=(3, 3, 9),
            num_classes=3,
            gamma=0.5,
            attn_mode=config.attn_mode,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_val_uf1, best_state, best_epoch, wait = -1.0, None, 0, 0
        hist = {'train_loss': [], 'val_loss': [], 'val_uf1': []}

        for epoch in range(1, epochs + 1):
            model.train()
            tr_loss = 0.0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * x.size(0)
            tr_loss /= max(len(train_dl.dataset), 1)

            # Validate: model selection uses validation UF1 only (test subject is never seen here).
            model.eval()
            v_loss, v_pred, v_gt = 0.0, [], []
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    v_loss += loss_fn(out, y).item() * x.size(0)
                    v_pred += torch.max(out, 1)[1].cpu().tolist()
                    v_gt += y.cpu().tolist()
            v_loss /= max(len(val_dl.dataset), 1)
            v_uf1, _ = recognition_evaluation(v_gt, v_pred)
            v_uf1 = float(v_uf1) if v_uf1 != '' else 0.0

            hist['train_loss'].append(tr_loss)
            hist['val_loss'].append(v_loss)
            hist['val_uf1'].append(v_uf1)
            if v_uf1 > best_val_uf1 + min_delta:
                best_val_uf1, best_epoch, wait = v_uf1, epoch, 0
                best_state = deepcopy(model.state_dict())
            else:
                wait += 1
            print('[Epoch %d] train_loss=%.4f val_loss=%.4f val_uf1=%.4f (best_epoch=%d wait=%d)'
                  % (epoch, tr_loss, v_loss, v_uf1, best_epoch, wait))
            if wait >= patience:
                print('  early stop at epoch %d: no val-UF1 improvement for %d epochs' % (epoch, patience))
                break

        # Test once, with the best-validation checkpoint.
        model.load_state_dict(best_state)
        model.eval()
        te_pred, te_gt = [], []
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                te_pred += torch.max(out, 1)[1].cpu().tolist()
                te_gt += y.cpu().tolist()

        # Same _acc.txt layout as before so calculate_all_results.py / aggregate_results.py still read it.
        matrix = [[int(p), int(g)] for p, g in zip(te_pred, te_gt)]
        test_acc = float(np.mean([p == g for p, g in zip(te_pred, te_gt)])) if te_gt else 0.0
        with open(os.path.join(results_dir, str(n_subName) + '_acc.txt'), 'a') as f:
            f.write('best epoach: ' + str(best_epoch) + '\n' + 'best acc: ' + str(test_acc)
                    + '\n' + 'matrix_acc: ' + str(matrix) + '\n')

        hist['best_epoch'] = best_epoch
        histories[n_subName] = hist
        total_pred += te_pred
        total_gt += te_gt
        UF1, UAR = recognition_evaluation(total_gt, total_pred)
        print('Subject %s done: n=%d best_epoch=%d | pooled UF1=%s UAR=%s'
              % (n_subName, len(te_gt), best_epoch, str(UF1), str(UAR)))

    with open(os.path.join(results_dir, 'history.json'), 'w') as f:
        json.dump(histories, f)

    print('Final Evaluation:')
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    print('pooled UF1=%s UAR=%s over n=%d' % (str(UF1), str(UAR), len(total_gt)))
    print('Total Time Taken:', time.time() - t)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=strtobool, default=True)
    parser.add_argument('--attn_mode', type=str, default='shared',
                        choices=['shared', 'dual', 'shared_matched'],
                        help="Phase attention: shared (CausalNet baseline), dual "
                             "(per-phase params, the contribution), or shared_matched (capacity control).")
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--epochs', type=int, default=200, help="Max epochs per fold.")
    parser.add_argument('--patience', type=int, default=40,
                        help="Early stop after this many epochs without val-UF1 improvement.")
    parser.add_argument('--run_name', type=str, default='',
                        help="Results subfolder name. Defaults to '<attn_mode>_seed<seed>'.")
    config = parser.parse_args()
    main(config)
