__all__ = ["model"]

from icevision.imports import *
from icevision.utils import *
from timm import create_model


def model(
    model_name,
    pretrained=False,
    checkpoint_path="",
    scriptable=None,
    exportable=None,
    no_jit=None,
    **kwargs
) -> nn.Module:
    """Creates the Ross Wightman `timm` model specified by `model_name`.

    Models implemented by Ross Wightman can be found in the original repository
    [here](https://github.com/rwightman/pytorch-image-models).

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load after model is initialized
        scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
        exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific

    # Returns
        A PyTorch model.
    """
    model = create_model(
        model_name=model_name,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        scriptable=scriptable,
        exportable=exportable,
        no_jit=exportable,
        **kwargs)
    
    # TODO: Break down param groups for backbone
    # def param_groups_fn(model: nn.Module) -> List[List[nn.Parameter]]:
    #     unwrapped = unwrap_bench(model)

    #     layers = [
    #         unwrapped.backbone,
    #         unwrapped.fpn,
    #         nn.Sequential(unwrapped.class_net, unwrapped.box_net),
    #     ]
    #     param_groups = [list(layer.parameters()) for layer in layers]
    #     check_all_model_params_in_groups2(model, param_groups)

    #     return param_groups

    # model.param_groups = MethodType(param_groups_fn, model)

    return model
