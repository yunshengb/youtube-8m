import math

import models
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim
import models
import tensorflow as tf
import numpy as np

FLAGS = flags.FLAGS
train = True

'''
Class name post-fixed with name, e.g. 'MyModel_mturkeli'.
Must have local_cmd and remote_cmd defined in the class.
'''

class MyModel_mturkeli(models.BaseModel):

    # local_cmd = getLocalCmd('train', 'train', 'train')
    @staticmethod
    def getRemoteCmd():
        if train:
            return '\t'.join(('BUCKET_NAME=gs://muratturkeli93_yt8m_train_bucket;\n',
                              'JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S);\n',
                              '/Users/muratturkeli/Downloads/google-cloud-sdk/bin/gcloud --verbosity=debug ml-engine jobs \\\n',
                              'submit training $JOB_NAME \\\n',
                              '--package-path=src \\\n',
                              '--module-name=src.train \\\n',
                              '--staging-bucket=$BUCKET_NAME \\\n',
                              '--region=us-east1 \\\n',
                              '--config=src/cloudml-gpu.yaml \\\n',
                              '-- --train_data_pattern="gs://youtube_8m_new_new_video/train/train*.tfrecord" \\\n',
                              '--train_dir=$BUCKET_NAME/MyModel_mturkeliSave \\\n',
                              '--frame_features=False \\\n',
                              '--model=MyModel_mturkeli \\\n',
                              '--feature_names="mean_rgb, mean_audio, std_rgb, std_audio" \\\n',
                              '--feature_sizes="1024, 128, 1024, 128" \\\n',
                              '--batch_size=128 \\\n',
                              '--moe_num_mixtures=7 \\\n',
                              '--base_learning_rate=0.05 \\\n'
                              ))
        else:
            return '\t'.join(('BUCKET_NAME=gs://muratturkeli93_yt8m_train_bucket;\n',
                              'JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S);\n',
                              '/Users/muratturkeli/Downloads/google-cloud-sdk/bin/gcloud --verbosity=debug ml-engine jobs \\\n',
                              'submit training $JOB_NAME \\\n',
                              '--package-path=src \\\n',
                              '--module-name=src.inference \\\n',
                              '--staging-bucket=$BUCKET_NAME \\\n',
                              '--region=us-east1 \\\n',
                              '--config=src/cloudml-gpu.yaml \\\n',
                              '-- --input_data_pattern="gs://youtube8m-ml-us-east1/1/video_level/test/test*.tfrecord" \\\n',
                              '--train_dir=$BUCKET_NAME/MyModel_mturkeliSave \\\n',
                              '--frame_features=False \\\n',
                              '--model=MyModel_mturkeli \\\n',
                              '--feature_names="mean_rgb, mean_audio, std_rgb, std_audio" \\\n',
                              '--feature_sizes="1024, 128, 1024, 128" \\\n',
                              '--batch_size=128 \\\n',
                              '--moe_num_mixtures=7 \\\n',
                              '--output_file=$BUCKET_NAME/MyModel_mturkeliSave/predictions.csv \\\n'
                              ))




    @staticmethod
    def getLocalCmd():
        if train:
            return '\t'.join(('python src/train.py \\\n',
                              '--train_data_pattern="data/video_aug/train*.tfrecord" \\\n',
                              '--train_dir=model/MyModel_mturkeliSave \\\n',
                              '--frame_features=False \\\n',
                              '--model=MyModel_mturkeli \\\n',
                              '--feature_names="mean_rgb, mean_audio, std_rgb, std_audio" \\\n',
                              '--feature_sizes="1024, 128, 1024, 128" \\\n',
                              '--batch_size=16 \\\n',
                              '--moe_num_mixtures=7 \\\n',
                              '--base_learning_rate=0.05 \\\n'))
        else:
            return '\t'.join(('python src/eval.py \\\n',
                              '--eval_data_pattern="data/video_new/validate*.tfrecord" \\\n',
                              '--train_dir=model/MyModel_mturkeliSave \\\n',
                              '--frame_features=False \\\n',
                              '--model=MyModel_mturkeli \\\n',
                              '--feature_names="mean_rgb, mean_audio, std_rgb, std_audio" \\\n',
                              '--feature_sizes="1024, 128, 1024, 128" \\\n',
                              '--batch_size=16 \\\n',
                              '--moe_num_mixtures=17 \\\n',
                              '--base_learning_rate=0.05 \\\n'))




    def my_gradient(self, x):
        return np.gradient(x, axis=1)


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

        print("feature size")
        print(feature_size)



        split0, split1, split2, split3 = tf.split(model_input, [1024, 128, 1024, 128], 1)

        rgb_mean_gradient = tf.py_func(self.my_gradient, [split0], [tf.float32])
        rgb_mean_gradient_r = tf.reshape(rgb_mean_gradient[0], [-1, 1024])

        audio_mean_gradient = tf.py_func(self.my_gradient, [split1], [tf.float32])
        audio_mean_gradient_r = tf.reshape(audio_mean_gradient[0], [-1, 128])

        rgb_std_gradient = tf.py_func(self.my_gradient, [split2], [tf.float32])
        rgb_std_gradient_r = tf.reshape(rgb_std_gradient[0], [-1, 1024])

        audio_std_gradient = tf.py_func(self.my_gradient, [split3], [tf.float32])
        audio_std_gradient_r = tf.reshape(audio_std_gradient[0], [-1, 128])

        new_input = tf.concat([model_input, rgb_mean_gradient_r], 1)
        new_input = tf.concat([new_input, audio_mean_gradient_r], 1)
        new_input = tf.concat([new_input, rgb_std_gradient_r], 1)
        new_input = tf.concat([new_input, audio_std_gradient_r], 1)


        # gradient = tf.py_func(self.my_gradient, [split0], [tf.float32])
        #
        #
        # gradient_reshaped = tf.reshape(gradient[0], [-1, 1024])
        #
        #
        # print(tf.rank(gradient_reshaped))
        # print(gradient_reshaped.get_shape())
        #
        # new_input = tf.concat([model_input, gradient_reshaped], 1)

        print(new_input.get_shape())

        gate_activations = slim.fully_connected(
            new_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            new_input,
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