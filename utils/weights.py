import torch

# Load from weights
def load_from_weights(model, weights, logger=None):
    if logger:
        logger.info("Loading weights from "+weights)
    else:
        print("Loading weights from "+weights)

    # Get weights from saved file and check in model
    ckpt = torch.load(weights, map_location='cpu')
    if 'model' in ckpt.keys():
        ckpt = ckpt['model']
    if 'teacher' in ckpt.keys():
        ckpt = ckpt['teacher']

    model_dict = model.state_dict()
    
    # Change the names of the keys
    loaded_dict = {k.replace("module.", "").replace("backbone.", "").replace("features.", ""): v for k, v in ckpt.items()}

    # Look in the models dicts the keys that match
    pretrained_dict = {}
    weights_ignored = []
    weights_loaded = []
    for k, v in model_dict.items():
        k_loaded = k.replace("module.", "").replace("backbone.", "").replace("features.", "")
        if k_loaded in loaded_dict.keys():
            match_size = (v.shape==loaded_dict[k_loaded].shape)
            if match_size:
                pretrained_dict[k] = loaded_dict[k_loaded]
                weights_loaded.append(k_loaded)
            else:
                weights_ignored.append(k)

    expdata = "  ".join(["{}".format(k) for k in weights_ignored])
    if logger:
        logger.info('Weights not found in loaded model: '+expdata)
        logger.info('----------------------------------')
    else:
        print('Weights not found in loaded model: '+expdata)
        print('----------------------------------')

    weights_not_used = []
    for k, v in loaded_dict.items():
        if k not in weights_loaded:
            weights_not_used.append(k)
    expdata = "  ".join(["{}".format(k) for k in weights_not_used])
    if logger:
        logger.info('Weights not used from loaded model: '+expdata)
        logger.info('----------------------------------')
    else:
        print('Weights not used from loaded model: '+expdata)
        print('----------------------------------')

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if logger:
        logger.info("Done loading pretrained weights")
    else:
        print("Done loading pretrained weights")

    return model


