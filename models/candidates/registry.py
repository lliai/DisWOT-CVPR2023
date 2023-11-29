_model_entrypoints = {}


def register_model(fn):
    _model_entrypoints[fn.__name__] = fn
    return fn


def model_entrypoints(model_name):
    return _model_entrypoints[model_name]


def is_model(model_name):
    return model_name in _model_entrypoints
