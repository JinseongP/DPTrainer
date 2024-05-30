import warnings

import torch

from .. import RobModel
from .. import load_model
from ._parse_notion import get_link_by_id


def load_pretrained(model_name, save_dir="./"):
    flag = None
    flag_for_db = ""
    if "_Best" in model_name:
        model_name = model_name.replace("_Best", "")
        flag = "_Best"
        flag_for_db = ""
    elif "_Last" in model_name:
        model_name = model_name.replace("_Last", "")
        flag = "_Last"
        flag_for_db = "(Last)"
    else:
        pass

    urls = get_link_by_id(model_name, flag_for_db)
    if len(urls.keys()) > 1:
        if flag is None:
            warnings.warn("There are two options[Last, Best], but the best model returned.")
            flag = "_Best"
        state_dict = torch.hub.load_state_dict_from_url(urls[model_name+flag+".pth"],
                                                        progress=True, model_dir=save_dir,
                                                        file_name=model_name+flag+".pth",
                                                        map_location='cpu')
    else:
        if flag is not None:
            warnings.warn("There exist only one pretrained verion.")
        state_dict = torch.hub.load_state_dict_from_url(list(urls.values())[0],
                                                        progress=True, model_dir=save_dir,
                                                        file_name=model_name+".pth",
                                                        map_location='cpu')
    return _construct_rmodel_from_dict(model_name, state_dict)


def _construct_rmodel_from_dict(model_name, state_dict):
    n_classes = state_dict['rmodel']['n_classes'].item()
    model = load_model(model_name=model_name.split("_")[1],
                       n_classes=n_classes)
    if 'norm.mean' in state_dict.keys():
        mean = state_dict['rmodel']['norm.mean'].numpy()
        std = state_dict['rmodel']['norm.std'].numpy()
        rmodel = RobModel(model, n_classes=n_classes,
                          normalize={'mean':mean, 'std':std})
        rmodel.load_state_dict_auto(state_dict['rmodel'])
    else:
        rmodel = RobModel(model, n_classes=n_classes)
        rmodel.load_state_dict_auto(state_dict['rmodel'])
    return rmodel
