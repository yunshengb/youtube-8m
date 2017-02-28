model = 'LstmModel'
option = 't'
machine = 'remote'

'''
LogisticModel
MoeModel
FrameLevelLogisticModel
DbofModel
LstmModel
'''

def main():
    option_msg = 'option must be "t" for train, ' \
                 '"e" for evaluate, or "i" for infer'
    if machine == 'local':
        if option == 't':
            execute('rm -rf model/{}'.format(getModelPath()))
            execute(getLocalCmd('train', 'train', 'train'))
        elif option == 'e':
            execute(getLocalCmd('eval', 'eval', 'validate'))
        elif option == 'i':
            execute(getLocalCmd('inference', 'input', 'test', \
                                 '--output_file=model/{}/predictions.csv' \
                                 .format(getModelPath())))
        else:
            print option_msg
    elif machine == 'remote':
        if option == 't':
            execute(getRemoteCmd('train', 'train', 'train'))
        elif option == 'e':
            execute(getRemoteCmd('eval', 'eval', 'validate'))
        elif option == 'i':
            execute(getRemoteCmd('inference', 'input', 'test', \
                                 '--output_file=$BUCKET_NAME/{}/predictions.csv' \
                                 .format(getModelPath())))
        else:
            print option_msg
    else:
        print 'machine must be "local" or "remote"'

def getLocalCmd(option, data_pattern, tfrecord, output_file=''):
    return '''python src/{}.py \\
    --{}_data_pattern="data/{}/{}*.tfrecord" \\
    --train_dir=model/{} \\
    {} {}'''.format \
        (option, data_pattern, 'frame' if isFrameLevel() else 'video', \
         tfrecord, getModelPath(), getModelParams(), \
         output_file)

def getRemoteCmd(option, data_pattern, tfrecord, output_file=''):
    return '''BUCKET_NAME=gs://${{USER}}_yt8m_train_bucket;
    JOB_NAME=yt8m_{}_$(date +%Y%m%d_%H%M%S);
    gcloud --verbosity=debug beta ml jobs \\
    submit training $JOB_NAME \\
    --package-path=src --module-name=src.{} \\
    --staging-bucket=$BUCKET_NAME --region=us-east1 \\
    --config=src/cloudml-gpu.yaml \\
    -- --{}_data_pattern="gs://youtube8m-ml-us-east1/1/{}_level/{}*.tfrecord" \\
    --train_dir=$BUCKET_NAME/{} \\
    {} {}'''.format \
        (option, option, data_pattern, 'frame' if isFrameLevel() else 'video', \
         tfrecord, tfrecord, getModelPath(), \
         getModelParams(), output_file)

def getModelParams():
    model_param = '--model={}'.format(model)
    frame_level = isFrameLevel()
    feature_prefix = 'mean_' if not frame_level else ''
    return '''--frame_features={} \\
    --feature_names="{}rgb, {}audio" \\
    --feature_sizes="1024, 128" \\
    --batch_size=512 \\
    {}'''.format(frame_level, feature_prefix, feature_prefix , model_param)

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
        print 'Cannot find model', model
        exit(1)
    return frame_level

def getModelNames(file):
    import src.video_level_models
    import src.frame_level_models
    import sys, inspect
    rtn = []
    for name, obj in inspect.getmembers(sys.modules['src.{}'.format(file)]):
        if inspect.isclass(obj):
            rtn.append(name)
    return rtn

def getModelPath():
    return model + 'Save'

def execute(cmd):
    print '-'*10, cmd
    from os import system
    system(cmd)

if __name__ == '__main__':
    main()
