import numpy as np
import tensorflow as tf

class SequenceSpliter:

    """A class that tranforms a sequence into a supervised learning problem"""
    def __init__(self, lookback, look_ahead, step=1):
        self.lookback = lookback
        self.look_ahead = look_ahead
        self.step = step

    def fit(self, X):
        return X

    def transform(self, x):
        X, y = [], []
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        for i in range(0, len(x), self.step):
            # find the end of this pattern
            end_ix = i + self.lookback
            out_end_idx = end_ix + self.look_ahead
            # check if we are beyond the dataset
            if out_end_idx > len(x):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = x[i:end_ix, :], x[end_ix:out_end_idx, :]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def fit_transform(self, X):
        X = self.fit(X)
        x, y = self.transform(X)
        return x, y


'''
 ' Huber loss.
 ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
 ' https://en.wikipedia.org/wiki/Huber_loss
'''
def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)

'''
 ' Same as above but returns the mean loss.
'''
def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
  return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))




