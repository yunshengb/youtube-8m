from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow import gfile
from tensorflow import logging
from multiprocessing import Queue, Pool
from os import getpid
from sklearn.preprocessing import normalize


input_data_pattern = 'gs://youtube8m-ml-us-east1/1/frame_level/train/train*.tfrecord'


def main():
    logging.set_verbosity(tf.logging.INFO)
    files = gfile.Glob(input_data_pattern)
    q = Queue()
    for file in files:
        q.put(file)
    logging.info('Put ' + str(len(files)) + ' files to the queue')
    Pool(3, worker_main, (q,))


def worker_main(q):
    logging.info('Worker ' + str(getpid()) + ' starts')
    try:
        i = 0
        while True:
            file = q.get(False)
            logging.info('Worker ' + str(getpid()) + ' reads from ' + file)
            generate_video_level_record(file, 'gs://youtube_8m_video/' + file.split('/')[-1])
            logging.info('Worker ' + str(getpid()) + ' has processed ' + str(i) + ' files')
    except Queue.Empty:
        logging.info('Worker ' + str(getpid()) + ' done')


def video_level_record_check(input_file):
    vid_ids = []
    labels = []
    mean_rgb = []
    mean_audio = []
    for example in tf.python_io.tf_record_iterator(input_file):
        tf_example = tf.train.Example.FromString(example)
        vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
        labels.append(tf_example.features.feature['labels'].int64_list.value)
        mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)
        mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)
    print('Number of videos in this tfrecord: ',len(mean_rgb))
    print('First video feature length',len(mean_rgb[0]))
    print('First 20 features of the first youtube video (',vid_ids[0],')')
    print(mean_rgb[0][:20])


def generate_video_level_record(input_file, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    logging.info('Worker ' + str(getpid()) + ' writes to ' + str(output_file))
    for example in tf.python_io.tf_record_iterator(input_file):
        tf_seq_example = tf.train.SequenceExample.FromString(example)
        tf_example = tf.train.Example.FromString(example)
        n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)
        sess = tf.InteractiveSession()
        rgb_frame = []
        audio_frame = []
        # iterate through frames.
        for i in range(n_frames):
            rgb_frame.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['rgb'].feature[
                    i].bytes_list.value[0], tf.uint8)
                , tf.float32).eval())
            audio_frame.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['audio'].feature[
                    i].bytes_list.value[0], tf.uint8)
                , tf.float32).eval())
        sess.close()
        # Calculate mean, std, 1st-5th features.
        rgb_mat = pad(np.array(rgb_frame).T, (1024, 300)) # (1024, 300)
        audio_mat = pad(np.array(audio_frame).T, (128, 300)) # (128, 300)
        rgb_stats_mat = get_stats_mat(rgb_mat) # (7, 1024)
        audio_stats_mat = get_stats_mat(rgb_mat)  # (7, 1024)
        feature = {
            'video_id': tf_example.features.feature['video_id'],
            'labels': tf_example.features.feature['labels'],
            'mean_rgb': float_feat(rgb_stats_mat[0]),
            'mean_audio': float_feat(audio_stats_mat[0]),
            'std_rgb': float_feat(rgb_stats_mat[1]),
            'std_audio': float_feat(audio_stats_mat[1]),
            '1st_rgb': float_feat(rgb_stats_mat[2]),
            '1st_audio': float_feat(audio_stats_mat[2]),
            '2nd_rgb': float_feat(rgb_stats_mat[3]),
            '2nd_audio': float_feat(audio_stats_mat[3]),
            '3rd_rgb': float_feat(rgb_stats_mat[4]),
            '3rd_audio': float_feat(audio_stats_mat[4]),
            '4th_rgb': float_feat(rgb_stats_mat[5]),
            '4th_audio': float_feat(audio_stats_mat[5]),
            '5th_rgb': float_feat(rgb_stats_mat[6]),
            '5th_audio': float_feat(audio_stats_mat[6])
        }
        # Write to tfrecord.
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())


def pad(a, s):
    p = np.zeros(s)
    p[:a.shape[0],:a.shape[1]] = a
    return p


def get_stats_mat(a):
    s = np.sort(a, axis=1)
    return normal(np.array((np.mean(a, axis=1),
                            np.std(rgb_mat, axis=1),
                            s[:,-1],
                            s[:,-2],
                            s[:,-3],
                            s[:,-4],
                            s[:,-5])))


def normal(a):
    a = a - a.mean(axis=1, keepdims=True)
    a = a / a.std(axis=1, keepdims=True)
    a = normalize(a, axis=1, norm='l2')
    return a * 10 # so it's closer to the provided video level feature values


def float_feat(value_list):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


if __name__ == '__main__':
    main()


# Fun area:
# x = np.array([[1, 2, 3, 0], [6, 5, 0, 4], [7, 8, 9, 0]])
# print(x)
# a1 = np.mean(x, axis=1)
# a2 = np.std(x, axis=1)
# print(a1)
# print(a2)
# print(a1 / np.linalg.norm(a1))
# print(normal(np.array((a1, a2))))
# x.sort(axis=1)
# print(x)
# print(x[:,-1])
# print(x[:,-2])
# print(x[:,-3])
# print(x[:,-4])
