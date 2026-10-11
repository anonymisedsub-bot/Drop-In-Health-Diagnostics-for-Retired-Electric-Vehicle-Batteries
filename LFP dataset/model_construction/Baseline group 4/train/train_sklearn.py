"""Configured tree regressors and the preserved point/quantile evaluation protocol."""


import numpy as np
from utils import summary_metrics, flatten_channel_time, denorm_y
from joblib import dump


def get_cgb(cfg, loss_params=None):
    from catboost import CatBoostRegressor
    params = cfg.ML_CONFIG['cgb'].copy()
    params.update(loss_params or {'loss_function': 'RMSE'})
    return CatBoostRegressor(**params)


def get_lgbm(cfg, loss_params=None):
    from lightgbm import LGBMRegressor
    params = cfg.ML_CONFIG['lgbm'].copy()
    params.update(loss_params or {'objective': 'regression_l2'})
    return LGBMRegressor(**params)


class MLQuantileWrapper:
    """
    Train q50 with squared error and other quantiles with pinball loss.
    """
    def __init__(self, name, base_model_fn, quantiles, cfg):
        self.name = name
        self.quantiles = quantiles
        self.models = {}
        
        for q in quantiles:
            # Set loss parameters according to the model type.
            if name == 'cgb' and q == 0.5:
                loss_params = {'loss_function': 'RMSE'}
            elif name == 'cgb':
                loss_params = {'loss_function': f'Quantile:alpha={q}'}
            elif name == 'lgbm' and q == 0.5:
                loss_params = {'objective': 'regression_l2'}
            elif name == 'lgbm':
                loss_params = {'objective': 'quantile', 'alpha': q}
            else:
                raise ValueError(f"Model {name} is not supported by MLQuantileWrapper")
            
            self.models[q] = base_model_fn(cfg=cfg, loss_params=loss_params)

    def fit(self, X, Y):
        Y = Y.ravel()
        print(f"[ML:{self.name}] Fitting {len(self.quantiles)} models for quantiles: {self.quantiles}...")
        for q, model in self.models.items():
            model.fit(X, Y)
        return self
    
    def predict(self, X):
        preds = []
        for q in self.quantiles:
            p = self.models[q].predict(X).reshape(-1, 1)
            preds.append(p)
        return np.hstack(preds) # (N, 3)
    
    def predict_quantiles(self, X, quantiles=None):
        if quantiles is None:
            quantiles = self.quantiles
        
        preds = []
        for q in quantiles:
            if q in self.models:
                p = self.models[q].predict(X).reshape(-1, 1)
                preds.append(p)
            else:
                available_q = np.array(self.quantiles)
                closest_q = available_q[np.argmin(np.abs(available_q - q))]
                p = self.models[closest_q].predict(X).reshape(-1, 1)
                preds.append(p)
                print(f"Warning: Using quantile {closest_q} for requested {q}")
        
        return np.hstack(preds)
    
    def predict_mean_std(self, X):
        preds = self.predict(X)
        mean_pred = preds[:, 1:2]
        q10 = preds[:, 0:1]
        q90 = preds[:, 2:3]
        std_pred = (q90 - q10) / (2 * 1.2816)
        return mean_pred, std_pred


def build_ml_models(cfg):
    """The configured padded-input quantile tree factories."""
    quantile = cfg.ML_CONFIG['quantile']
    return {
        name: (lambda name=name, factory=factory: MLQuantileWrapper(
            name, factory, quantiles=quantile['quantiles'], cfg=cfg))
        for name, factory in {'cgb': get_cgb, 'lgbm': get_lgbm}.items()
        if name in quantile['enabled_models']
    }


def eval_ml_model(model, Xtr, Ytr, Xval, Yval, Xte, Yte, norm=None, quantiles=[0.5]):
    """Evaluate the given ML model on train/val/test subsets."""
    if not isinstance(model, MLQuantileWrapper):
        raise TypeError(f"Unsupported padding ML model type: {type(model)}")

    Xtr2, Xval2, Xte2 = map(flatten_channel_time, (Xtr, Xval, Xte))

    # Fit the model if it has not already been trained.
    if (Ytr is not None) and (Ytr.shape[0] > 0):
        model.fit(Xtr2, Ytr)

    out = {}
    for split, (X2, Y) in {"train": (Xtr2, Ytr), "val": (Xval2, Yval), "test": (Xte2, Yte)}.items():
        if Y is None or Y.shape[0] == 0:
            out[split] = {}
            continue

        # Obtain predictions.
        yp_n = model.predict(X2).reshape(-1, len(quantiles))

        # Reverse normalization and evaluate.
        if norm is not None:
            Y_den = denorm_y(Y, norm)
            YP_den = denorm_y(yp_n, norm)
            out[split] = summary_metrics(Y_den, YP_den, quantiles=quantiles)
            out[split]["y_true_den"] = Y_den
            out[split]["y_pred_den"] = YP_den
        else:
            out[split] = summary_metrics(Y, yp_n, quantiles=quantiles)
            out[split]["y_true_den"] = Y
            out[split]["y_pred_den"] = yp_n
        out[split]["pred"] = yp_n
            
    return out


def save_ml(model, path):
    dump(model, path)
