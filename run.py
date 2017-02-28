model = 'video_level_logistic_model'
option = 't'
machine = 'remote'

def execute(cmd):
    print '-'*10, cmd
    from os import system
    system(cmd)

def getRemoteCmd(option, data_pattern, tfrecord, output_file=''):
    cmd = '''BUCKET_NAME=gs://${{USER}}_yt8m_train_bucket;
    JOB_NAME=yt8m_{}_$(date +%Y%m%d_%H%M%S);
    gcloud --verbosity=debug beta ml jobs \\
    submit training $JOB_NAME \\
    --package-path=src --module-name=src.{} \\
    --staging-bucket=$BUCKET_NAME --region=us-east1 \\
    --config=youtube-8m/src/cloudml-gpu.yaml \\
    -- --{}_data_pattern='
    "gs://youtube8m-ml-us-east1/1/video_level/{}/{}*.tfrecord" \\
    --train_dir=$BUCKET_NAME/{}'
    {}'''.format \
        (option, option, data_pattern, tfrecord, tfrecord, model, output_file)
    return cmd

option_msg = 'option must be "t" for train, "e" for evaluate, or "i" for infer'

if machine == 'local':
    if option == 't':
        execute('cd src && python train.py '
                '--train_data_pattern="../data/train*.tfrecord" '
                '--train_dir=../model/{}'.format(model))
    elif option == 'e':
        execute('cd src && python eval.py '
                '--eval_data_pattern="../data/validate*.tfrecord"'
                ' --train_dir=../model/{} --run_once=True'.format(model))
    elif option == 'i':
        execute('cd src && python inference.py '
                '--output_file=../model/{}/predictions.csv '
                '--input_data_pattern="../data/test*.tfrecord"'
                ' --train_dir=../model/{}'.format(model, model))
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
                             .format(model)))
    else:
        print option_msg
else:
    print 'machine must be "local" or "remote"'
