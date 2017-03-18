import models

'''
Class name post-fixed with name, e.g. 'MyModel_ykohan'.
Must have local_cmd and remote_cmd defined in the class.
'''

class MyModel_ykohan(models.BaseModel):

    local_cmd = 'python src/trian.py ...'
    remote_cmd = 'gcloud ...'

    def create_model():
        pass
