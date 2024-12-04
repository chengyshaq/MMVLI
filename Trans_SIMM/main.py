# -*- coding: UTF-8 -*-
from config import *
from model.model.main_train import main_simm_model_train
from model.utilities.common_tools import *
from model.model.main_prediction import *
from model.utilities.ml_metrics import *
import os
import math
from scipy import stats


print('*' * 30)
print('ML Loss coefficient:\t', loss_coefficient['ML_loss'] )
print('has Gan:\t', model_args['has_GAN'])
print('has Comm ML Loss:\t', model_args['has_comm_ML_Loss'])
if model_args['has_GAN']:
    print('Gan coefficient:\t', loss_coefficient['GAN_loss'])
if model_args['has_comm_ML_Loss']:
    print('Comm ML Loss coefficient:\t', loss_coefficient['comm_ML_loss'])
print('has orthogonal regularization:\t', model_args['has_orthogonal_regularization'])
if model_args['has_orthogonal_regularization']:
    print('orthogonal_regularization coefficient:\t', loss_coefficient['orthogonal_regularization'])
    print('orthogonal_regularization type:\t', model_args['regularization_type'])
print('dataset:\t', DATA_SET_NAME)
print('comm feature num:\t', ARGS['comm_feature_nums'])
print('optimizer:\t Adam')
print('*' * 30)

features, labels, idx_list = load_mat_data_v2(os.path.join(DATA_ROOT, DATA_SET_NAME), True)
label_num = np.size(labels, 1)
metrics_result_list = []
avg_metrics = {}
for fold in range(Fold_numbers):
    TEST_SPLIT_INDEX = fold
    print('-' * 50 + '\n' + 'Fold: %s' % fold + '\n')
    train_features, train_labels, test_features, test_labels = split_data_set_by_idx(features, labels,
                                                                              idx_list, TEST_SPLIT_INDEX)

    loss = main_simm_model_train(train_features, train_labels, ARGS,
                                 loss_coefficient, model_args, WEIGHT_DECAY, fold)

    epoch_to_predict = 0
    model_state_path = os.path.join(ARGS['model_save_dir'],
                                    'fold' + str(fold)+'_' + 'epoch' + str(epoch_to_predict + 3) + '.pth')

    outputs = main_prediction(features=test_features,
                              label_nums=label_num,
                              model_state_path=model_state_path,
                              model_args=model_args,
                              args=ARGS)

    pre_labels = np.array(outputs > 0.5, dtype=np.int)
    true_labels = np.array(test_labels, dtype=np.int)
    metrics_name, metrics_result = all_metrics(outputs, pre_labels, true_labels)

    metrics_result_list.append(metrics_result)

print("------------summary--------------")
result=np.array(metrics_result_list)
result=np.around(result,4)
np.savetxt('corel5k_results.csv', result, delimiter=',')
mean=np.mean(result,axis=0)
print_mean=np.around(mean,4)
std=np.std(result,axis=0) 
print_std=np.around(std, 4)
ste = std / math.sqrt(5)
print_ste = np.around(ste, 4)
for k in range(0,7):
    # writeline = [metrics_name[k], mean[k], std[k], ste[k]]
    # csv_write.writerow(writeline)
    print("{metric}:\t{mean}\t{std}\t{ste}".format(metric=metrics_name[k],mean=print_mean[k],std=print_std[k], ste=print_ste[k]))

confidence_hl = stats.t.interval(0.95, df=4, loc=mean[0], scale=ste[0])
print(confidence_hl)


