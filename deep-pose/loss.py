import tensorflow as tf


class PixelMSE(tf.keras.losses.Loss):
    def __init__(self, pos_val=0.5):
        super(PixelMSE, self).__init__()
        self.pos_val = pos_val
        self.pos_loss = tf.keras.losses.MeanSquaredError()
        self.neg_loss = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred, pixel_weights=None):
        pos_indices = tf.where(y_true > 0.0)
        neg_indices = tf.where(y_true == 0.0)

        y_true_pos = tf.gather_nd(y_true, pos_indices)
        y_pred_pos = tf.gather_nd(y_pred, pos_indices)

        y_true_neg = tf.gather_nd(y_true, neg_indices)
        y_pred_neg = tf.gather_nd(y_pred, neg_indices)

        mse_pos = self.pos_loss(y_true_pos, y_pred_pos)
        mse_neg = self.neg_loss(y_true_neg, y_pred_neg)

        mse = self.pos_val * mse_pos + (1 - self.pos_val) * mse_neg
        return mse