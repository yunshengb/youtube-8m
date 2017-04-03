import math

import models
import utils

from tensorflow import flags
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

    @staticmethod
    def getRemoteCmd():
        return '\t'.join(('BUCKET_NAME=gs://muratturkeli93_yt8m_train_bucket2;\n',
                          'JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S);\n',
                          '/Users/muratturkeli/Downloads/google-cloud-sdk/bin/gcloud --verbosity=debug ml-engine jobs \\\n',
                          'submit training $JOB_NAME \\\n',
                          '--package-path=src \\\n',
                          '--module-name=src.train \\\n',
                          '--staging-bucket=$BUCKET_NAME \\\n',
                          '--region=us-east1 \\\n',
                          '--config=src/cloudml-gpu.yaml \\\n',
                          '-- --train_data_pattern="gs://youtube8m-ml-us-east1/1/video_level/train/train*.tfrecord" \\\n',
                          '--train_dir=$BUCKET_NAME/MyModel_mturkeliSave \\\n',
                          '--frame_features=False \\\n',
                          '--model=MyModel_mturkeli \\\n',
                          '--feature_names="mean_rgb, mean_audio" \\\n',
                          '--feature_sizes="1024, 128" \\\n',
                          '--moe_num_mixtures=7 \\\n'
                          ))


    @staticmethod
    def getLocalCmd():

        return '\t'.join(('python src/train.py \\\n',
                          '--train_data_pattern="data/video/train*.tfrecord" \\\n',
                          '--train_dir=model/MuratModelSave \\\n',
                          '--frame_features=False \\\n',
                          '--model=MyModel_mturkeli \\\n',
                          '--feature_names="mean_rgb, mean_audio" \\\n',
                          '--feature_sizes="1024, 128" \\\n',
                          '--batch_size=1024 \\\n',
                          '--moe_num_mixtures=7 \\\n'))




    def create_model(self, model_input, vocab_size, num_mixtures=None, l2_penalty=1e-8, **unused_params):


        """
            Args:
              model_input: 'batch_size' x 'num_features' matrix of input features.
              vocab_size: The number of classes in the dataset.
              num_mixtures: The number of mixtures (excluding a dummy 'expert' that
                always predicts the non-existence of an entity).
              l2_penalty: How much to penalize the squared magnitudes of parameter
                values.

        """

        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

        feature_size = model_input.get_shape().as_list()[1]
        reshaped_input = tf.reshape(model_input, [-1, -1, feature_size, 1])

        # Conv layers
        convLayer1 = slim.repeat(
            reshaped_input,
            1,
            slim.conv2d,
            64,
            [3, 3],
            scope = 'conv1'
        )
        convLayer2 = slim.repeat(
            convLayer1,
            1,
            slim.conv2d,
            128,
            [3, 3],
            scope='conv2'
        )

        # Pooling layer
        poolLayer1 = slim.max_pool2d(
            convLayer2,
            [2, 2],
            scope = 'pool1'
        )

        convLayer3 = slim.repeat(
            poolLayer1,
            1,
            slim.conv2d,
            64,
            [3, 3],
            scope = 'conv3'
        )

        convLayer4 = slim.repeat(
            convLayer3,
            1,
            slim.conv2d,
            128,
            [3, 3],
            scope = 'conv4'
        )

        poolLayer2 = slim.max_pool2d(
            convLayer4,
            [2, 2],
            scope='pool2'
        )


        gate_activations = slim.fully_connected(
            poolLayer2,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            poolLayer2,
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