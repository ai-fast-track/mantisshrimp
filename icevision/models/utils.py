__all__ = [
    "filter_params",
    "unfreeze",
    "freeze",
    "transform_dl",
    "common_build_batch",
    "_predict_dl",
]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.parsers import *
from icevision.data.dataset import Dataset


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def filter_params(
    module: nn.Module, bn: bool = True, only_trainable=False
) -> Generator:
    """Yields the trainable parameters of a given module.

    Args:
        module: A given module
        bn: If False, don't return batch norm layers

    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not isinstance(module, BN_TYPES) or bn:
            for param in module.parameters():
                if not only_trainable or param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(
                module=child, bn=bn, only_trainable=only_trainable
            ):
                yield param


def unfreeze(params):
    for p in params:
        p.requires_grad = True


def freeze(params):
    for p in params:
        p.requires_grad = False


def transform_dl(dataset, build_batch, batch_tfms=None, **dataloader_kwargs):
    collate_fn = partial(build_batch, batch_tfms=batch_tfms)
    return DataLoader(dataset=dataset, collate_fn=collate_fn, **dataloader_kwargs)


def common_build_batch(records: Sequence[RecordType], batch_tfms=None):
    if batch_tfms is not None:
        records = batch_tfms(records)

    return records


@torch.no_grad()
def _predict_dl(
    predict_fn,
    model: nn.Module,
    infer_dl: DataLoader,
    show_pbar: bool = True,
    **predict_kwargs,
):
    all_preds, all_samples = [], []
    for batch, samples in pbar(infer_dl, show=show_pbar):
        preds = predict_fn(model=model, batch=batch, **predict_kwargs)

        all_samples.extend(samples)
        all_preds.extend(preds)

    return all_samples, all_preds


def get_dataloaders(
    model_type,
    records_list: List[List[dict]],
    tfms_list: List[Transform],
    size,
    presize,
    batch_tfms=None,
    batch_size=16,
    num_workers=4,
    **dataloader_kwargs,
):
    """
    Creates and returns datasets and dataloaders:

    # Arguments
        model_type: can be one of these values: faster_rcnn, retinanet, mask_rcnn, efficientdet
        records: A list of records ->  [train_records, valid_records].
            Both train_records, valid_records are of type List[dict]
        tfms: List of Transforms to be applied to each dataset: [train_tfms, valid_tfms].
            Both train_tfms and valid_tfms are of type Transform
        size: The final size of the image.
        presizing: Rescale the image before applying other transfroms.
        batch_size: batch size.
        num_workers: number of workers.

    # Return
        - datasets: List containing train_ds and valid_ds -> [train_ds, valid_ds]
        - dataloaders: List containing train_dl and valid_dl -> [train_dl, valid_dl]
    """

    ds = []
    dls = []

    # Datasets
    train_ds = Dataset(records_list[0], tfms_list[0])
    valid_ds = Dataset(records_list[1], tfms_list[1])

    # Dataloaders
    train_dl = model_type.train_dl(
        train_ds,
        batch_tfms=batch_tfms,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        **dataloader_kwargs,
    )
    valid_dl = model_type.valid_dl(
        valid_ds,
        batch_tfms=batch_tfms,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        **dataloader_kwargs,
    )

    ds.append(train_ds)
    ds.append(valid_ds)

    dls.append(train_dl)
    dls.append(valid_dl)

    return dls
