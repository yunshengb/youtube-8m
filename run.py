#!/usr/bin/env python
option = 't'
model = 'MoeModel'
local = False
batch = 64
features = [('mean_rgb', 1024), ('std_rgb', 1024), ('1st_rgb', 1024),
            ('2nd_rgb', 1024), ('3rd_rgb', 1024), ('4th_rgb', 1024),
            ('5th_rgb', 1024),
            ('mean_audio', 128), ('std_audio', 128), ('1st_audio', 128),
            ('2nd_audio', 128), ('3rd_audio', 128), ('4th_audio', 128),
            ('5th_rgb', 1024)]
extra = [('moe_num_mixtures', 4)]
save = '_new_all_4exp_gpu'


'''
('moe_num_mixtures', 7)
('base_learning_rate', 0.0001)
('lstm_layers', 5)
'''

'''
option: 't' | 'e' | 'i'
model: 'LogisticModel' | 'MoeModel' | 'FrameLevelLogisticModel' | 'DbofModel' |
       'LstmModel' | 'MyModel_mturkeli'
batch: None (default batch size) | [integer]
local: True | False
extra (parameter): None (no extra parameter) | [([name], [value]), ...]
'''


import src.video_level_models
import src.frame_level_models
import src.yba_models
import src.tkohan_models
import src.mturkeli_models


def main():
    msg = 'option must be "t" for train, "e" for evaluate, or "i" for infer'
    if local:
        if option == 't':
            execute('rm -rf model/{}'.format(getModelPath()))
            execute(getLocalCmd('train', 'train', 'train'))
        elif option == 'e':
            execute(getLocalCmd('eval', 'eval', 'validate'))
        elif option == 'i':
            execute(getLocalCmd('inference', 'input', 'test',
                                '--output_file=model/{}/predictions.csv'
                                .format(getModelPath())))
        else:
            print msg
    else:
        if option == 't':
            execute(getRemoteCmd('train', 'train', 'train'))
        elif option == 'e':
            execute(getRemoteCmd('eval', 'eval', 'validate'))
        elif option == 'i':
            execute(getRemoteCmd('inference', 'input', 'test',
                                 '--output_file=$BUCKET_NAME/{}/predictions.csv'
                                 .format(getModelPath())))
        else:
            print msg


def getLocalCmd(option, data_pattern, tfrecord, output_file=''):
    if isOurModel():
        return isOurModel().getLocalCmd()
    f_type = 'frame' if isFrameLevel() else 'video'
    params = synthesizeParam(
        [('%s_data_pattern' % data_pattern,
          '"data/%s/%s*.tfrecord"' % (f_type, tfrecord)),
         ('train_dir', 'model/%s' % getModelPath())])
    return '\t'.join(('python src/%s.py \\\n' % option, params,
                      getModelParams(), output_file))


def getRemoteCmd(option, data_pattern, tfrecord, output_file=''):
    if isOurModel():
        return isOurModel().getRemoteCmd()
    f_type = 'frame' if isFrameLevel() else 'video'
    params = [('package-path', 'src'),
              ('module-name', 'src.%s' % option),
              ('staging-bucket', '$BUCKET_NAME'),
              ('region', 'us-east1'),
              ('config', 'src/cloudml-gpu.yaml'),
              (' --%s_data_pattern' % data_pattern,
               '"gs://youtube_8m_augment_video/%s/%s*.tfrecord"' %
               (tfrecord, tfrecord)),
              ('train_dir', '$BUCKET_NAME/%s' % getModelPath())]
    return '\t'.join(('BUCKET_NAME=gs://${USER}_yt8m_train_bucket;\n',
                      'JOB_NAME=yt8m_{}_$(date +%Y%m%d_%H%M%S);\n'.format(option),
                      'gcloud --verbosity=debug ml-engine jobs \\\n',
                      'submit training $JOB_NAME \\\n', synthesizeParam(params),
                      getModelParams(),
                      output_file))


def isOurModel():
    yba_models = getModelNames('yba_models')
    mturkeli_models = getModelNames('mturkeli_models')
    tkohan_models = getModelNames('tkohan_models')
    if model in yba_models:
        user = 'yba'
    elif model in mturkeli_models:
        user = 'mturkeli'
    elif model in tkohan_models:
        user = 'tkohan'
    else:
        return False
    assert(user in model)
    return eval('src.%s_models.%s' % (user, model))


def getModelParams(frame_level_ = False):
    if isOurModel():
        frame_level = frame_level_
    else:
        frame_level = isFrameLevel()
    prefix = 'mean_' if not frame_level else ''
    params = [('frame_features', frame_level),
              ('model', model),
              ('feature_names', '"%s"' % ', '.join([i[0] for i in features])),
              ('feature_sizes', '"%s"' % ', '.join([str(i[1]) for i in features]))]
    if batch:
        params.append(('batch_size', batch))
    if extra:
        for i in extra:
            params.append(i)
    return synthesizeParam(params)


def synthesizeParam(params):
    return '\t'.join(['--%s=%s \\\n' % (p, v) for p, v in params])


def isFrameLevel():
    video_level_models = getModelNames('video_level_models')
    frame_level_models = getModelNames('frame_level_models')
    if model in video_level_models:
        if model in frame_level_models:
            print 'Duplicate model name', model, 'in video and frame models'
            exit(1)
        else:
            frame_level = False
    elif model in frame_level_models:
        frame_level = True
    else:
        print 'Unrecognized model', model
        exit(1)
    return frame_level


def getModelNames(file):
    import sys, inspect
    rtn = []
    for name, obj in inspect.getmembers(sys.modules['src.{}'.format(file)]):
        if inspect.isclass(obj):
            rtn.append(name)
    return rtn


def getModelPath():
    return model + save


def execute(cmd):
    print '-' * 10, cmd
    from os import system
    system(cmd)


if __name__ == '__main__':
    main()
