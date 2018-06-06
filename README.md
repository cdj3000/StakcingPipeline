# StakcingPipeline
one layer stacking : Catboost, Lightboost, RandomForest, Xgboost, Extratree

the model accepts given params

model_dict = {'xgboost': 
                        {'model', object, 'params': params}},
              'catboost':
                        {'model', object, 'params': params},
              'rft':
                        {'model', object, 'params': params},
              'et':
                        {'model', object, 'params': params},
              'lightgbm':
                        {'model', object, 'params': params}}
