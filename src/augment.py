from __future__ import print_function
from utils import Dequantize
import tensorflow as tf
import numpy as np
from tensorflow import gfile
from tensorflow import logging
from multiprocessing import Queue, Process
from Queue import Empty
from os import getpid
from sklearn.preprocessing import normalize
from readers import resize_axis
from time import time
from os import path
from random import shuffle


# Modify.
type = 'train'
input_data_pattern = 'gs://youtube8m-ml-us-east1/1/frame_level/%s/%s*.tfrecord' \
                     % (type, type)
output_data_dir = 'gs://youtube_8m_new_video/%s/' % type
local = False

# Local testing.
if local:
    local_dir = '/Users/yba/Documents/U/EECS_351/youtube-8m/data/'
    input_data_pattern = local_dir + 'frame/train*.tfrecord'
    output_data_dir = local_dir + 'new_video/'

num_classes = 4716
feature_sizes = [1024, 128]
feature_names = ['rgb', 'audio']
max_frames = 300


def main():
    logging.set_verbosity(tf.logging.INFO)
    files = gfile.Glob(input_data_pattern)
    # Local checking.
    if local:
        video_level_record_check(local_dir + 'new_video/%sa0.tfrecord' % type)
    q = Queue()
    num_files = 0
    shuffle(files)
    for file in files:
        q.put(file)
        num_files += 1
    logging.info('Main put ' + str(num_files) + ' files to the queue')
    ps = []
    for i in range(3): # 3 workers: tested on Google Cloud Platform large_model
        p = Process(target=worker_main, args=(q,))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()


def worker_main(q):
    logging.info('Worker ' + str(getpid()) + ' starts')
    i = 0
    try:
        while True:
            file = q.get(False)
            if not need_process(file, get_output_file(file)):
                logging.info('Worker %s skipping %s' % (getpid(), file))
                continue
            logging.info('Worker %s reads from %s' % (getpid(), file))
            t = time()
            num_vid = generate_video_level_record(file, get_output_file(file))
            i += 1
            logging.info('Worker %s processed %sth file with %s videos in '
                         'in %.2f sec' \
                         % (getpid(), i, num_vid, (time()-t)))
    except Empty:
        logging.info('Worker %s done' % getpid())


def need_process(input_file, output_file):
    if not output_file in gfile.Glob(output_data_dir + '*.tfrecord'):
        return True
    #if gfile.GFile(output_file, "rb").size() == 0:
    #    print(input_file, output_file, ' 0 size')
    #    return True
    #if get_num_video(input_file) != get_num_video(output_file):
    #    print(input_file, output_file, ' insufficient videos')
    #    return True
    return False


def get_num_video(input_file):
    num_vid = 0
    for _ in tf.python_io.tf_record_iterator(input_file):
        num_vid += 1
    return num_vid


def get_output_file(input_file_full_path):
    return output_data_dir + input_file_full_path.split('/')[-1]


def video_level_record_check(input_file):
    vid_ids = []
    labels = []
    mean_rgb = []
    first_audio = []
    fifth_audio = []
    if not path.isfile(input_file):
        print('Input file doesn\'t exist')
        return
    for example in tf.python_io.tf_record_iterator(input_file):
        tf_example = tf.train.Example.FromString(example)
        vid_ids.append(tf_example.features.feature['video_id'].bytes_list.
                       value[0].decode(encoding='UTF-8'))
        labels.append(tf_example.features.feature['labels'].int64_list.value)
        mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.
                        value)
        first_audio.append(tf_example.features.feature['1st_audio'].float_list.
                           value)
        fifth_audio.append(tf_example.features.feature['5th_audio'].float_list.
                           value)
    print('Number of videos in tfrecord', input_file, ':', len(mean_rgb))
    print('First video (', vid_ids[0], ')')
    print('labels')
    unique, counts = np.unique(labels[0], return_counts=True)
    print(dict(zip(unique, counts)))
    print('mean rgb features')
    print(mean_rgb[0][:20])
    print('1st audio features')
    print(first_audio[0][:20])
    print('5th audio features')
    print(fifth_audio[0][:20])


def generate_video_level_record(input_file, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    logging.info('Worker %s writes to %s' % (getpid(), output_file))
    num_vid = get_num_video(input_file)
    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer([input_file])
        feat = read_and_decode(filename_queue)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(num_vid):
            feat_rtn = sess.run([feat])
            process_video(feat_rtn[0], writer) # feat_rtn[0] is a dict
            if (i+1) % 100 == 0:
                logging.info('Worker ' + str(getpid()) + ' ' + str(i+1) + \
                             ' videos')
        coord.request_stop()
        coord.join(threads)
    return i+1 # number of videos processed


def process_video(d, writer):
    rgb_frame = d['rgb']
    audio_frame = d['audio']
    # Calculate mean, std, 1st-5th features.
    rgb_mat = np.array(rgb_frame).T
    audio_mat = np.array(audio_frame).T
    assert(rgb_mat.shape == (1024, 300))
    assert(audio_mat.shape == (128, 300))
    rgb_stats_mat = get_stats_mat(rgb_mat) # (7, 1024)
    audio_stats_mat = get_stats_mat(audio_mat)  # (7, 1024)
    feature = {
        'video_id': byte_feat([d['video_id']]), # note: [[...]] -> proper string
        'labels': int64_feat(d['labels']),
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


def get_stats_mat(a):
    s = np.sort(a, axis=1)
    return normal(np.array((np.mean(a, axis=1),
                            np.std(a, axis=1),
                            s[:,-1],
                            s[:,-2],
                            s[:,-3],
                            s[:,-4],
                            s[:,-5])))


def normal(a):
    a = a - a.mean(axis=1, keepdims=True)
    # * 10 so it's closer to the provided video level feature values.
    return normalize(a, axis=1, norm='l2') * 10


def byte_feat(value_list):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_list))


def int64_feat(value_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


def float_feat(value_list):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    contexts, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features={"video_id": tf.FixedLenFeature(
            [], tf.string),
            "labels": tf.VarLenFeature(tf.int64)},
        sequence_features={
            feature_name: tf.FixedLenSequenceFeature([], dtype=tf.string)
            for feature_name in feature_names
            })
    video_id = contexts["video_id"]
    labels = (tf.cast(
        tf.sparse_to_dense(contexts["labels"].values, (num_classes,), 1,
                           validate_indices=False),
        tf.int64))
    rtn = {}
    rtn['video_id'] = video_id
    rtn['labels'] = labels
    # Loads (potentially) different types of features.
    num_features = len(feature_names)
    assert num_features > 0, "No feature selected: feature_names is empty!"
    assert len(feature_names) == len(feature_sizes), \
        "length of feature_names (={}) != length of feature_sizes (={})". \
            format(len(feature_names), len(feature_sizes))
    num_frames = -1  # the number of frames in the video
    for feature_index in range(num_features):
        feature_name = feature_names[feature_index]
        feature_matrix, num_frames_in_this_feature = get_video_matrix(
            features[feature_name],
            feature_sizes[feature_index],
            max_frames,
            2,
            -2)
        if num_frames == -1:
            num_frames = num_frames_in_this_feature
        else:
            tf.assert_equal(num_frames, num_frames_in_this_feature)
        rtn[feature_name] = feature_matrix
    return rtn


def get_video_matrix(features,
                     feature_size,
                     max_frames,
                     max_quantized_value,
                     min_quantized_value):
    decoded_features = tf.reshape(
        tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
    feature_matrix = Dequantize(decoded_features,
                                max_quantized_value,
                                min_quantized_value)
    feature_matrix = resize_axis(feature_matrix, 0, max_frames)
    return feature_matrix, num_frames


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
