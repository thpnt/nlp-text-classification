import pandas as pd
import tensorflow as tf


# Define custom loss
weights = pd.Series([0.334298, 0.665702])
class WeightedCategoricalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, weights, name='weighted_categorical_crossentropy', **kwargs):
        super(WeightedCategoricalCrossEntropy, self).__init__()
        self.weights = tf.cast(weights, tf.float32)
        
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # Clip y_pred to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        weighted_losses = -self.weights * y_true * tf.math.log(y_pred)
        return tf.reduce_mean(tf.reduce_sum(weighted_losses, axis=1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "weights": self.weights,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)






# Define custom metrics
class PrecisionMulticlass(tf.keras.metrics.Metric):
    def __init__(self, name='precision', n_class=2, **kwargs):
        super(PrecisionMulticlass, self).__init__(name=name, **kwargs)
        self.precision = self.add_weight(
            shape=(n_class,),
            name='precision',
            initializer='zeros')
        self.n_class = n_class
        self.true_positives = self.add_weight(name='true_positives', shape=(self.n_class,), initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', shape=(self.n_class,), initializer='zeros')
        
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int64)
        y_pred = tf.cast(tf.one_hot(tf.argmax(y_pred, axis=1), self.n_class), tf.int64)
        
        for i in range(self.n_class):
            true_positive = tf.reduce_sum(y_true[:, i] * y_pred[:, i])
            false_positive = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true[:, i], 0), tf.equal(y_pred[:, i], 1)), tf.int64))
            
            index = [[i]]  # Index for the class we are updating
            self.true_positives.assign(tf.tensor_scatter_nd_add(self.true_positives, index, [true_positive]))
            self.false_positives.assign(tf.tensor_scatter_nd_add(self.false_positives, index, [false_positive]))
            
    def result(self):
        precision_per_class = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        return tf.reduce_mean(precision_per_class)
    
    def reset_state(self):
        self.true_positives.assign(tf.zeros(self.n_class))
        self.false_positives.assign(tf.zeros(self.n_class))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_class": self.n_class,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        

class RecallMulticlass(tf.keras.metrics.Metric):
    def __init__(self, name='recall', n_class=2, **kwargs):
        super(RecallMulticlass, self).__init__(name=name, **kwargs)
        self.recall = self.add_weight(
            shape=(n_class,),
            name='recall',
            initializer='zeros')
        self.n_class = n_class
        self.true_positives = self.add_weight(name='true_positives', shape=(self.n_class,), initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', shape=(self.n_class,), initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int64)
        y_pred = tf.cast(tf.one_hot(tf.argmax(y_pred, axis=1), self.n_class), tf.int64)
        
        for i in range(self.n_class):
            true_positive = tf.reduce_sum(y_true[:, i] * y_pred[:, i])
            false_negative = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true[:, i], 1), tf.equal(y_pred[:, i], 0)), tf.int64))
            
            index = [[i]]  # Index for the class we are updating
            self.true_positives.assign(tf.tensor_scatter_nd_add(self.true_positives, index, [true_positive]))
            self.false_negatives.assign(tf.tensor_scatter_nd_add(self.false_negatives, index, [false_negative]))
            
    def result(self):
        recall_per_class = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return tf.reduce_mean(recall_per_class)
    
    def reset_state(self):
        self.true_positives.assign(tf.zeros(self.n_class))
        self.false_negatives.assign(tf.zeros(self.n_class))
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_class": self.n_class,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
        
        
class F1ScoreMulticlass(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', n_class=2, **kwargs):
        super(F1ScoreMulticlass, self).__init__(name=name, **kwargs)
        self.n_class = n_class
        self.precision = PrecisionMulticlass(n_class=n_class)
        self.recall = RecallMulticlass(n_class=n_class)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1_score

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_class": self.n_class,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
        
            