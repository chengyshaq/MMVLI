# -*- coding: UTF-8 -*-

Fold_numbers = 5
DATA_ROOT = './'
DATA_SET_NAME = 'Yeast_10_fold_origin_2_views.mat'

TEST_SPLIT_INDEX = 1

ARGS = {}
ARGS['epoch'] = 10
ARGS['comm_feature_nums'] = 64
ARGS['show_epoch'] = 1
ARGS['epoch_used_for_final_result'] = 4
ARGS['model_save_epoch'] = 3
ARGS['model_save_dir'] = 'model_save_dir'

WEIGHT_DECAY = 1e-5

loss_coefficient = {}
loss_coefficient['ML_loss'] = 1.0
loss_coefficient['GAN_loss'] = 0.1
loss_coefficient['comm_ML_loss'] = 0.1
loss_coefficient['orthogonal_regularization'] = 0.001

zero_thread = 1e-30
model_args = {}
model_args['regularization_type'] = 'L1'
model_args['has_orthogonal_regularization'] = True if abs(loss_coefficient['orthogonal_regularization']) > zero_thread else False
model_args['has_GAN'] = True if abs(loss_coefficient['GAN_loss']) > zero_thread else False
model_args['has_comm_ML_Loss'] = True if abs(loss_coefficient['comm_ML_loss']) > zero_thread else False

