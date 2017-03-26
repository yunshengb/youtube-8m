import math

import models
import utils

from tensorflow import flags
from run import *
import tensorflow.contrib.slim as slim
import models
import tensorflow as tf

FLAGS = flags.FLAGS

'''
Class name post-fixed with name, e.g. 'MyModel_mturkeli'.
Must have local_cmd and remote_cmd defined in the class.
'''

class MyModel_mturkeli(models.BaseModel):

    # local_cmd = getLocalCmd('train', 'train', 'train')
    remote_cmd = 'gcloud ...'


    @staticmethod
    def getLocalCmd():
        f_type = 'video'
        params = synthesizeParam(
            [('train_dir', 'model/%s' % 'MyModel_mturkeli' + 'Save')])
        option = 'train'
        return '\t'.join(('python src/%s.py \\\n' % option, params,
                          getModelParams(False), ''))


    def create_model(self, model_input, vocab_size, num_mixtures, l2_penalty=1e-8, **unused_params):


        """
            Args:
              model_input: 'batch_size' x 'num_features' matrix of input features.
              vocab_size: The number of classes in the dataset.
              num_mixtures: The number of mixtures (excluding a dummy 'expert' that
                always predicts the non-existence of an entity).
              l2_penalty: How much to penalize the squared magnitudes of parameter
                values.

        """
        pass


        # Conv layer
        convLayer1 = tf.layers.conv2d(
            inputs = model_input,
            filter =  1,
            kernel_size = 1,
            padding = "same",
            activation=tf.nn.relu
        )

        # Pooling layer
        poolLayer1 = tf.layers.max_pooling2d(
            input = convLayer1,
            pool_size= [2, 2],
            strides = 2
        )

        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        return {"predictions": final_probabilities}