__all__ = ["learner"]

from icevision.imports import *
from icevision.engines.fastai import *
# from icevision.models.ross.timm.loss_fn import loss_fn
from icevision.models.ross.timm.fastai.callbacks import TIMMCallback


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    loss_func=None,
    **learner_kwargs,
):
    """Fastai `Learner` adapted for timm.

    # Arguments
        dls: `Sequence` of `DataLoaders` passed to the `Learner`.
        The first one will be used for training and the second for validation.
        model: The model to train.
        cbs: Optional `Sequence` of callbacks.
        **learner_kwargs: Keyword arguments that will be internally passed to `Learner`.

    # Returns
        A fastai `Learner`.
    """
    # cbs = [TIMMCallback()] + L(cbs)
    cbs = L(cbs)

    learn = adapted_fastai_learner(
        dls=dls,
        model=model,
        cbs=cbs,
        loss_func=loss_func,
        **learner_kwargs,
    )

    return learn
