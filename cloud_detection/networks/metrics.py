# Custom metrics for the auto-encoder

import tensorflow as tf


# True positive on hard masks
class BinaryTP(tf.keras.metrics.Metric):
   def __init__(self, name="binary_TP", **kwargs):
      super(BinaryTP, self).__init__(name=name, **kwargs)
      self.TP = self.add_weight(name="tp", initializer='zeros')
   
   def update_state(self, y_true, y_pred, sample_weight=None):
      y_pred = tf.math.round(y_pred)

      y_true = tf.cast(y_true, tf.bool)
      y_pred = tf.cast(y_pred, tf.bool)

      values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
      values = tf.cast(values, self.dtype)
      if not sample_weight == None:
         sample_weight = tf.cast(sample_weight, self.dtype)
         sample_weight = tf.broadcast_to(sample_weight, values.shape)
         values = tf.multiply(values, sample_weight)
      self.TP.assign_add(tf.reduce_sum(values))
   
   def result(self):
      return self.TP


# False positive on hard masks
class BinaryFP(tf.keras.metrics.Metric):
   def __init__(self, name="binary_FP", **kwargs):
      super(BinaryFP, self).__init__(name=name, **kwargs)
      self.FP = self.add_weight(name="fp", initializer='zeros')
   
   def update_state(self, y_true, y_pred, sample_weight=None):
      y_pred = tf.math.round(y_pred)

      y_true = tf.cast(y_true, tf.bool)
      y_pred = tf.cast(y_pred, tf.bool)

      values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
      values = tf.cast(values, self.dtype)
      if not sample_weight == None:
         sample_weight = tf.cast(sample_weight, self.dtype)
         sample_weight = tf.broadcast_to(sample_weight, values.shape)
         values = tf.multiply(values, sample_weight)
      self.FP.assign_add(tf.reduce_sum(values))
   
   def result(self):
      return self.FP


# True negative on hard masks
class BinaryTN(tf.keras.metrics.Metric):
   def __init__(self, name="binary_TN", **kwargs):
      super(BinaryTN, self).__init__(name=name, **kwargs)
      self.TN = self.add_weight(name="tn", initializer='zeros')
   
   def update_state(self, y_true, y_pred, sample_weight=None):
      y_pred = tf.math.round(y_pred)

      y_true = tf.cast(y_true, tf.bool)
      y_pred = tf.cast(y_pred, tf.bool)

      values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
      values = tf.cast(values, self.dtype)
      if not sample_weight == None:
         sample_weight = tf.cast(sample_weight, self.dtype)
         sample_weight = tf.broadcast_to(sample_weight, values.shape)
         values = tf.multiply(values, sample_weight)
      self.TN.assign_add(tf.reduce_sum(values))
   
   def result(self):
      return self.TN


# False negative on hard masks
class BinaryFN(tf.keras.metrics.Metric):
   def __init__(self, name="binary_FN", **kwargs):
      super(BinaryFN, self).__init__(name=name, **kwargs)
      self.FN = self.add_weight(name="fn", initializer='zeros')
   
   def update_state(self, y_true, y_pred, sample_weight=None):
      y_pred = tf.math.round(y_pred)

      y_true = tf.cast(y_true, tf.bool)
      y_pred = tf.cast(y_pred, tf.bool)

      values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
      values = tf.cast(values, self.dtype)
      if not sample_weight == None:
         sample_weight = tf.cast(sample_weight, self.dtype)
         sample_weight = tf.broadcast_to(sample_weight, values.shape)
         values = tf.multiply(values, sample_weight)
      self.FN.assign_add(tf.reduce_sum(values))

   def result(self):
      return self.FN