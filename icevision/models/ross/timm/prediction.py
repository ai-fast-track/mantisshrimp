__all__ = ["predict", "predict_dl", "convert_raw_predictions"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.models.utils import _predict_dl


@torch.no_grad()
def predict(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    detection_threshold: float = 0.5,
    device: Optional[torch.device] = None,
):
    device = device or model_device(model)
    imgs, img_info = batch
    imgs = imgs.to(device)
    # img_info = {k: v.to(device) for k, v in img_info.items()}

    # bench = DetBenchPredict(unwrap_bench(model))
    model = model.eval().to(device)

    # raw_preds = model(x=imgs, img_info=img_info)
    # return convert_raw_predictions(raw_preds, detection_threshold=detection_threshold)
    raw_preds = model(imgs)
    return raw_preds


def predict_dl(
    model: nn.Module, infer_dl: DataLoader, show_pbar: bool = True, **predict_kwargs
):
    return _predict_dl(
        predict_fn=predict,
        model=model,
        infer_dl=infer_dl,
        show_pbar=show_pbar,
        **predict_kwargs,
    )

 # TODO Handle both single and multi-label cases 
def convert_raw_predictions(
    raw_preds: torch.Tensor, detection_threshold: float
) -> List[dict]:

    raw_preds = raw_preds.detach().cpu().numpy()
    preds = []
    for raw_pred in raw_preds:
        
        pred = {
            "scores": raw_pred,
            "labels": list(range(len(raw_pred)))
        }
        preds.append(pred)

    return preds
