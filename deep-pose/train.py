import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from dataset import get_coco_dataset
from models import *
from plt_utils import plot_results
from loss import PixelMSE
from datetime import datetime
from tensorflow.keras.optimizers import Adam


def get_dataset(args, mode):
    if mode == 'train':
        data_dir = args.train_dir
        annotation_file = args.train_annotations
    elif mode == 'val':
        data_dir = args.val_dir
        annotation_file = args.val_annotations

    dataset = get_coco_dataset(data_dir, annotation_file)
    dataset = dataset.map(lambda x, y: (tf.image.resize(
        x, (args.image_size, args.image_size)), y))
    dataset = dataset.map(lambda x, y:
                          (tf.image.random_brightness(x, max_delta=0.25), y))
    dataset = dataset.map(
        lambda x, y: (tf.image.random_contrast(x, lower=0.8, upper=1.2), y))

    dataset = dataset.map(lambda x, y: random_flip(x, y))
    dataset = dataset.shuffle(100, reshuffle_each_iteration=True)
    dataset = dataset.batch(args.batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)
    return dataset


def random_flip(x, y):
    if tf.random.uniform_candidate_sampler(
        [[1, 2]], 2, 1, False, 2, seed=None, name=None)[0] > 0:
        x = tf.image.flip_left_right(tf.image.flip_up_down(x))
        y = tf.image.flip_left_right(tf.image.flip_up_down(y))
    return x, y


def train(train_ds, val_ds, args, **kwargs):
    print("Starting training...")

    model = get_model(args.model_name)(stacks=args.stacks).model
    print(model.summary())
    tf.keras.utils.plot_model(model,
                              './model.png',
                              show_shapes=True,
                              expand_nested=True)

    if args.warm_start:
        print("loading from checkpoint...")
        model.load_weights(args.last_ckpt)
        start_epoch = int(str(args.last_ckpt).split("/")[-1].split("_")[1]) + 1
        print("start epoch: ", start_epoch)
    else:
        start_epoch = 0

    optim = Adam(learning_rate=args.learning_rate)

    global_step = -1
    for epoch in range(start_epoch, start_epoch + args.epochs):
        criteria = PixelMSE()
        for idx, data in enumerate(train_ds):
            loss_avg = tf.keras.metrics.Mean()
            global_step += 1
            with tf.GradientTape() as tape:
                tape.watch(model.trainable_variables)
                iter_losses = []
                images = data[0]
                keypoints = data[1]

                res = model(images)
                if not isinstance(res, list):
                    res = [res]
                    # For #stacks > 1

                # Multiple losses for stacking
                for res_idx, op in enumerate(res):
                    iter_loss = criteria(keypoints, op)
                    if idx % 10 == 0:
                        tf.summary.scalar(f'loss/train_inter{str(res_idx)}', iter_loss, step=global_step)
                    iter_losses.append(iter_loss)

                total_loss = tf.reduce_sum(iter_losses)
                loss_avg.update_state(total_loss)
                gradients = tape.gradient(total_loss,
                                          model.trainable_variables)
                optim.apply_gradients(zip(gradients,
                                          model.trainable_variables))

                if not args.warm_start and epoch == 0:
                    if idx == 0:
                        tf.keras.backend.set_value(optim.lr, 2.5e-6)
                    if idx != 0 and idx % 100 == 0:
                        curr_lr = tf.keras.backend.get_value(optim.lr)
                        if curr_lr < 2.5e-4:
                            new_lr = curr_lr * 2 if curr_lr * 2 < 2.5e-4 else 2.5e-4
                            tf.keras.backend.set_value(optim.lr, new_lr)
                            print(f"Changed current learning rate to: {tf.keras.backend.get_value(optim.lr)}")

                if args.cycle_lr and idx % 700 == 0:
                    if not 'lr_idx' in kwargs.keys():
                        kwargs['lr_idx'] = 0
                        next_lr = kwargs['lr_schedule'][kwargs['lr_idx']]
                        kwargs['lr_idx'] = kwargs['lr_idx'] + 1
                    else:
                        next_lr = kwargs['lr_schedule'][kwargs['lr_idx']]
                        kwargs['lr_idx'] = kwargs['lr_idx'] + 1 if kwargs[
                            'lr_idx'] + 1 < len(kwargs['lr_schedule']) else 0

                    tf.keras.backend.set_value(optim.lr, next_lr)
                    print(
                        f"Changed current learning rate to: {tf.keras.backend.get_value(optim.lr)}"
                    )

                if idx % 10 == 0:
                    print(
                        f"epoch: {epoch + 1}, iter: {idx}, total loss: {loss_avg.result()}, predictions > 0.75 - 1.0: {tf.where((res[-1] >= 0.75) & (res[-1] <= 1.0)).shape}"
                    )
                    tf.summary.image('train_images/input_images',
                                     plot_results(images, keypoints),
                                     step=global_step,
                                     max_outputs=4)
                    tf.summary.image('train_images/prediction_keypoints',
                                     tf.reduce_max(keypoints,
                                                   axis=-1,
                                                   keepdims=True),
                                     step=global_step,
                                     max_outputs=4)
                    tf.summary.image('train_images/train_results',
                                     plot_results(images, res[-1]),
                                     step=global_step)

                    tf.summary.scalar('loss/train_loss',
                                      loss_avg.result(),
                                      step=global_step)
                    tf.summary.scalar('learning_rate',
                                      tf.keras.backend.get_value(optim.lr),
                                      step=global_step)
                    tf.summary.scalar('positives/train',
                                      tf.where((res[-1] >= 0.75)
                                               & (res[-1] <= 1.0)).shape[0],
                                      step=global_step)
                    loss_avg.reset_states()

                if idx % args.val_interval == 0 and idx != 0:
                    # run validation loop
                    print("Running validation...")
                    val_criteria = PixelMSE()
                    val_loss_avg = tf.keras.metrics.Mean()

                    for val_idx, data in enumerate(val_ds):
                        val_iter_losses = []
                        val_images = data[0]
                        val_keypoints = data[1]

                        val_res = model(val_images, training=False)
                        if not isinstance(val_res, list):
                            val_res = [val_res]

                        # Multiple losses for stacking
                        for v_res_idx, val_op in enumerate(val_res):
                            val_iter_loss = val_criteria(val_keypoints, val_op)
                            val_iter_losses.append(val_iter_loss)
                        val_total_loss = tf.reduce_sum(val_iter_losses)
                        val_loss_avg.update_state(val_total_loss)

                        # For testing
                        if val_idx == 0:
                            break

                    print(
                        f"epoch: {epoch + 1}, iter: {idx}, total val loss: {val_loss_avg.result()}, predictions > 0.75 - 1.0: {tf.where((val_res[-1] >= 0.75) & (val_res[-1] <= 1.0)).shape}"
                    )
                    tf.summary.scalar('loss/val_loss',
                                      val_loss_avg.result(),
                                      step=global_step)
                    tf.summary.image('val_images/val_results',
                                     plot_results(val_images, val_res[-1]),
                                     step=global_step)
                    tf.summary.scalar(
                        'positives/val',
                        tf.where((val_res[-1] >= 0.75)
                                 & (val_res[-1] <= 1.0)).shape[0],
                        step=global_step)

            # # For testing
            # if idx == 20:
            #     exit(1)

            if idx % 200 == 0 and idx != 0:
                model.save_weights(os.path.join(kwargs['ckpt_dir'],
                                                f'e_{str(epoch)}_i{str(idx)}_g{global_step}'),
                                   save_format='tf')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtrain',
                        dest='train_dir',
                        help="Train data directory",
                        type=str)
    parser.add_argument('--dval',
                        dest='val_dir',
                        help="Validation data directory",
                        type=str)
    parser.add_argument('--atrain',
                        dest='train_annotations',
                        help="Train annotation file",
                        type=str)
    parser.add_argument('--aval',
                        dest='val_annotations',
                        help="Val annotation file",
                        type=str)
    parser.add_argument('--batch',
                        type=int,
                        dest='batch_size',
                        default=12,
                        help="Batch size")
    parser.add_argument('--lr',
                        type=float,
                        dest='learning_rate',
                        default=2.5e-4,
                        help="learning rate")
    parser.add_argument(
        '--cycle_lr',
        dest='cycle_lr',
        default=0,
        type=int,
        help="Increases a smaller LR progressively to target LR")
    parser.add_argument('--model_name',
                        dest='model_name',
                        default='HGNet',
                        type=str,
                        help="Name of model")
    parser.add_argument('--stacks', dest='stacks', type=int, default=1, help="Number of HG stacks")
    parser.add_argument('--lrl',
                        dest='lr_low',
                        type=float,
                        default=2.5e-6,
                        help="Lower bound for cyclic LR")
    parser.add_argument('--lru',
                        dest='lr_high',
                        type=float,
                        default=2.5e-4,
                        help="Higher bound for cyclic LR")
    parser.add_argument('--cycle_interval',
                        dest='cycle_interval',
                        default=700,
                        type=int,
                        help="interval for cyclic LR")
    parser.add_argument('--image_hw',
                        dest='image_size',
                        type=int,
                        default=256,
                        help="Image height, width tuple")
    parser.add_argument('--val_interval',
                        dest='val_interval',
                        type=int,
                        default=10,
                        help="Validation interval iterations")
    parser.add_argument('--ckpt_dir',
                        dest='ckpt_dir',
                        type=str,
                        help="Checkpoint directory")
    parser.add_argument('--epochs',
                        dest='epochs',
                        type=int,
                        default=1,
                        help="Number of epochs")
    parser.add_argument('--warm_start',
                        dest='warm_start',
                        default=1,
                        type=int,
                        help="0=False, 1=True")
    parser.add_argument('--last_ckpt',
                        dest='last_ckpt',
                        type=str,
                        help="Checkpoint to start warm start from")
    parser.add_argument('--tb_dir',
                        dest='tb_dir',
                        type=str,
                        help="tensorboard directory")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    kwargs = {}
    experiment_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_lr{str(args.learning_rate)}_b{str(args.batch_size)}_s{str(args.stacks)}"

    tb_log = tf.summary.create_file_writer(os.path.join(args.tb_dir, experiment_name))

    ckpt_dir = os.path.join(args.ckpt_dir, experiment_name)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        kwargs['ckpt_dir'] = ckpt_dir

    if args.cycle_lr:
        step_size = (args.lr_high - args.lr_low) / 10
        lr_schedule1 = np.arange(args.lr_low, args.lr_high + step_size,
                                 step_size)
        lr_schedule2 = np.flip(lr_schedule1, axis=-1)
        lr_schedule = np.concatenate([lr_schedule1, lr_schedule2], axis=-1)
        print(f"LR schedule: {lr_schedule}")
        kwargs['lr_schedule'] = list(lr_schedule)

    train_ds = get_dataset(args, 'train')
    val_ds = get_dataset(args, 'val')

    with tb_log.as_default():
        train(train_ds, val_ds, args, **kwargs)
