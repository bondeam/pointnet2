'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
PARENT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(PARENT_DIR, 'util'))
import provider
import tf_util
import las_dataset
import log_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg_LAS', help='Model name [default: pointnet2_cls_ssg_LAS]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--n_augmentations', type=int, default=1, help='Number of augmentations option: 1-6 [default: 1]')
parser.add_argument('--rgb', action='store_true', default=False, help='Use of RGB channels: True/False [default: True]')
parser.add_argument('--intensity', action='store_true', default=False, help='Use of intensity channels: True/False [default: False]')
parser.add_argument('--xyz', action='store_true', default=False, help='Add global XYZ channels: True/False [default: False]')
parser.add_argument('--xyzonly', action='store_true', default=False, help='Only use global XYZ channels: True/False [default: False]')
parser.add_argument('--trajectory', action='store_true', default=False, help='Add trajectory reference channels: True/False [default: False]')
parser.add_argument('--return_info', action='store_true', default=False, help='Add return num and num returns reference channels: True/False [default: False]')
parser.add_argument('--grid_size', type=int, default=8, help='Size of grid option: 4,8,16 [default: 8]')

#parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
N_AUGMENTATIONS = FLAGS.n_augmentations


XYZ = FLAGS.xyz #added
XYZONLY = FLAGS.xyzonly
RGB = FLAGS.rgb
INTENSITY = FLAGS.intensity
TRAJECTORY = FLAGS.trajectory
RETURN_INFO = FLAGS.return_info
GRID_SIZE = FLAGS.grid_size
hash_col = 'hash'+str(GRID_SIZE).zfill(2)
#hash_col = 'hash'

POINT_DIM = 3 #added
if XYZ: POINT_DIM += 3
if RGB: POINT_DIM += 3
if INTENSITY: POINT_DIM += 1
if TRAJECTORY: POINT_DIM += 2
if RETURN_INFO: POINT_DIM += 2
    
data_channels = ['XN','YN','ZN']
if XYZONLY: data_channels = ['X', 'Y', 'Z']
if INTENSITY: data_channels = ['intensity'] + data_channels
if RGB: data_channels = ['r', 'g', 'b'] + data_channels
if XYZ: data_channels = ['X', 'Y', 'Z'] + data_channels
if TRAJECTORY: data_channels.extend(['d_traj', 'h_traj'])
if RETURN_INFO: data_channels.extend(['return_num', 'num_returns'])

    
log = log_util.Log(FLAGS)
log.copy_file('train_custom.py')


MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
log.copy_file(MODEL_FILE)
#LOG_DIR = FLAGS.log_dir
#if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
#os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
#os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
#LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
#LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 4096
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

#NUM_CLASSES = 40
NUM_CLASSES = 2

# train/test split
data_dir = '/root/data'
file = 'las_hashed'
#datasplit_path = '/root/data/test/data_split.json'
df, train_hashes, test_hashes, validation_hashes = las_dataset.load_data_memory(data_dir, file, hash_col, log)
TRAIN_DATASET = las_dataset.Las_Dataset(df, train_hashes, num_point=NUM_POINT, \
                                        batch_size=BATCH_SIZE, shuffled=True, train=True, \
                                        data_channels=data_channels,N_AUGMENTATIONS=N_AUGMENTATIONS, \
                                        segment=False)
TRAIN_DATASET.balance_data()
TEST_DATASET = las_dataset.Las_Dataset(df, test_hashes, num_point=NUM_POINT, \
                                        batch_size=BATCH_SIZE, shuffled=True, train=False,\
                                        data_channels=data_channels,N_AUGMENTATIONS=1, \
                                        segment=False)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            #pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,POINT_DIM)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            MODEL.get_loss(pred, labels_pl, end_points)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses + [total_loss]:
                tf.summary.scalar(l.op.name, l)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(log.LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(log.LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log.out('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(log.LOG_DIR, "model.ckpt"))
                log.out("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    log.out('----')
    log.out(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TRAIN_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]

    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch()
        #batch_data = provider.random_point_dropout(batch_data)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        if (batch_idx+1)%100 == 0:
            print('batch %03d:  mean loss: %f' % (batch_idx+1,loss_sum / 50))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
        batch_idx += 1

    log.out('Train mean loss: %f' % (loss_sum / 50))
    log.out('Train accuracy: %f' % (total_correct / float(total_seen)))
    TRAIN_DATASET.reset()
        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_predict_class = [0 for _ in range(NUM_CLASSES)]
    total_conmat = [[0 for _ in range(NUM_CLASSES)] for __ in range(NUM_CLASSES)] 

    
    log.out(str(datetime.now()))
    log.out('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch()
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)
            total_predict_class[pred_val[i]] += 1
            total_conmat[l][pred_val[i]] += 1
    eval_mean_loss = loss_sum / float(batch_idx)
    eval_accuracy = total_correct / float(total_seen)
    precision_by_class = np.array(total_correct_class)/np.array(total_predict_class)
    recall_by_class = np.array(total_correct_class)/np.array(total_seen_class)
    log.out('eval mean loss: %f' % (eval_mean_loss))
    log.out('eval accuracy: %f'% (eval_accuracy))
    log.out('eval avg class acc: %f' % (np.mean(recall_by_class,dtype=np.float)))
    col_names = ['Correct','Seen','Precision','Recall']
    header = list(range(NUM_CLASSES))
    tbl = [total_correct_class, total_seen_class,precision_by_class,recall_by_class]
    log.printTable (tbl, header=header,col_names=col_names)
    log.out('')
    log.printTable(total_conmat,header=header,col_names=header,left_edges=False)
              
    EPOCH_CNT += 1

    TEST_DATASET.reset()
    return eval_accuracy


if __name__ == "__main__":
    log.out('pid: %s'%(str(os.getpid())))
    train()
    log.close()
