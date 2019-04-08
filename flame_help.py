from flame.build import Build
from flame.predict import Predict
from flame.manage import *
import pandas as pd
import json


def build_model(end, parameters, file):
    '''
    Instantiate flame Build class assigning
    passed parameter values (in a dict). Then
    build the model for passed file in passed 
    endpoint
    '''
    build = Build(end)
    # Set model parameters
    for parameter in parameters.keys():
        build.param.setVal(parameter, parameters[parameter])
    build.run(file)
    build.param.update_file(end)


def get_predictions(models_frame, name_col, version_col, file):
    """
    Get prediction of a set of compounds for a given list of models.
    Endpoint and version column names needed. Returns a dataframe
    with predictions for each endpoint as columns
    """
    model_predictions = pd.DataFrame()
    for model, version in zip(models_frame[name_col],
                              models_frame[version_col]):
        pred =  Predict(model, version)
        results = pred.run(file)
        results = json.loads(results[1])
        c0 = np.asarray(results['c0'])
        c1 = np.asarray(results['c1'])
        final = []
        for val, val2 in zip(c0, c1):
            if val and not val2:
                final.append(0)
            elif val2 and not val:
                final.append(1)
            else:
                final.append(2)
        model_predictions[model + '-' + str(version)] = final
    return model_predictions


def get_cross_val_stats(models):
    """ 
        Get cross-val statistics for models given 
        in a list of lists [[endpoint, version]]
    """
    lista = []
    columns = ['version', 'model']
    for index, model in enumerate(models):
        results = json.loads(action_results(model[0],model[1])[1])['model_valid_info']
        values = model
        if index == 0:
            for el in results:
                columns.append(el[0])
        for el in results:
            values.append(el[-1])
        lista.append(values)
    training = pd.DataFrame(lista, columns=columns)
    return(training)

def get_quality(results):
    lista = []
    results = json.loads(results[1])
    results = results['external-validation']
    for el in results:
        lista.append(el[2])
    return lista