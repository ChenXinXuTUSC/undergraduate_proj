import utils

from . import simpleunet

MODELS = []


def add_models(module):
    MODELS.extend([getattr(module, attr) for attr in dir(module) if 'Net' in attr or 'MLP' in attr])
add_models(simpleunet)

def load_model(name):
    '''
    Creates and returns an instance of the model given its class name.
    '''
    # Find the model class from its name
    mdict = {model.__name__: model for model in MODELS}
    if name not in mdict:
        utils.log_info(f'Invalid model index. You put {name}. Options are:')
        # Display a list of valid model names
        for model in MODELS:
            print('\t* {}'.format(model.__name__))
        return None
    NetClass = mdict[name]

    return NetClass