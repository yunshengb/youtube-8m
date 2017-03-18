import models

'''
Class name post-fixed with name, e.g. 'MyModel_yba'.
Must have local_cmd and remote_cmd defined in the class.
'''

class MyModel_yba(models.BaseModel):

    local_cmd = 'python src/trian.py ...'
    remote_cmd = 'gcloud ...'

    def create_model():
        pass
