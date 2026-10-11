# train/train_sklearn.py
import numpy as np
from utils import summary_metrics, flatten_channel_time, denorm_y
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
from sklearn.base import is_regressor
from sklearn.exceptions import NotFittedError


class XGBQuantileObjective:
    """
    Custom XGBoost quantile loss:
    L_q(y, y_hat) = q * (y - y_hat)      (y >= y_hat)
                  = (q - 1) * (y - y_hat) (y < y_hat)

    For sklearn.XGBRegressor, objective must be a callable(labels, preds),
    where labels and preds are both numpy.ndarray objects.
    """
    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    def __call__(self, y_true, y_pred):
        import numpy as np
        # y_true, y_pred: 1D numpy arrays
        err = y_true - y_pred  # e = y - ŷ

        # First derivative of quantile loss with respect to the prediction:
        # e > 0 (ŷ < y)  => grad = -alpha
        # e <= 0 (ŷ >= y)=> grad = 1 - alpha
        grad = np.where(err > 0.0, -self.alpha, 1.0 - self.alpha)

        # Approximate the Hessian with a constant for numerical stability.
        hess = np.ones_like(err, dtype=np.float32)
        return grad, hess


def get_xgb(cfg, loss_params=None):
    """Build XGBoost with the retained q50-MSE/quantile objectives."""
    from xgboost import XGBRegressor
    params = dict(cfg.ML_CONFIG['xgb'])
    if loss_params:
        if loss_params.get('objective') == 'quantile' and loss_params.get('alpha') is not None:
            params['objective'] = XGBQuantileObjective(loss_params['alpha'])
        else:
            params.update(loss_params)
    else:
        params.setdefault('objective', 'reg:squarederror')
    return XGBRegressor(**params)


def get_cgb(cfg, loss_params=None):
    """Build CatBoost from the exact settings in config.py."""
    from catboost import CatBoostRegressor
    params = dict(cfg.ML_CONFIG['cgb'])
    params.update(loss_params or {'loss_function': 'RMSE'})
    return CatBoostRegressor(**params)

def get_lgbm(cfg, loss_params=None):
    """Build LightGBM from the exact settings in config.py."""
    from lightgbm import LGBMRegressor
    params = dict(cfg.ML_CONFIG['lgbm'])
    params.update(loss_params or {'objective': 'regression_l2'})
    return LGBMRegressor(**params)


class MLQuantileWrapper:
    """
    Train q50 with squared error and other output slots with pinball loss.
    """
    def __init__(self, name, base_model_fn, quantiles, cfg):
        """
        Initialize the quantile regression wrapper.

        Args:
            name (str): Name of a regression model supported by this group.
            base_model_fn (callable): Base model constructor.
            quantiles (list[float]): Quantiles to predict.
            cfg (dict): Model configuration.

        Raises:
            ValueError: Raised for an unsupported model name.
        """
        self.name = name
        self.quantiles = quantiles
        self.models = {}
        # self.cfg = cfg
        
        for q in quantiles:
            # Select loss parameters for the model type.
            if q == 0.5:
                # Keep the point-output slot; train it with squared error.
                # CatBoost names its native squared-error objective "RMSE".
                point_losses = {
                    'cgb': {'loss_function': 'RMSE'},
                    'lgbm': {'objective': 'regression_l2'},
                    'xgb': {'objective': 'reg:squarederror'},
                }
                if name not in point_losses:
                    raise ValueError(f"Unsupported point regression model: {name}")
                loss_params = point_losses[name]
            elif name == 'cgb':
                loss_params = {'loss_function': f'Quantile:alpha={q}'}
            elif name == 'lgbm':
                loss_params = {'objective': 'quantile', 'alpha': q}
            elif name == 'xgb':
                # get_xgb handles objective='quantile' together with alpha.
                loss_params = {'objective': 'quantile', 'alpha': q}                
            else:
                raise ValueError(f"MLQuantileWrapper does not support model {name}")
            
            self.models[q] = base_model_fn(cfg=cfg, loss_params=loss_params)

    def fit(self, X, Y):
        Y = Y.ravel()
        print(f"[ML:{self.name}] Fitting {len(self.quantiles)} output models (q50 MSE; others pinball): {self.quantiles}...")
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
    """Publish only the active group models, using the supplied configuration."""
    quantiles = cfg.ML_CONFIG['quantile']['quantiles']
    enabled = cfg.ML_CONFIG['quantile']['enabled_models']
    factories = {'cgb': get_cgb, 'lgbm': get_lgbm, 'xgb': get_xgb}
    models = {}
    for name, factory in factories.items():
        if name in enabled:
            models[name] = lambda name=name, factory=factory: MLQuantileWrapper(name, factory, quantiles, cfg)
        else:
            models[name] = lambda factory=factory: factory(cfg)
    models['rf'] = lambda: RandomForestRegressor(**cfg.ML_CONFIG['rf'])
    return models

# Evaluation helpers
def get_rf_prediction_with_std(model, X):
    """Get mean and standard-deviation predictions from an RF model."""
    try:
        preds_per_tree = [tree.predict(X) for tree in model.estimators_]
    except NotFittedError:
        return model.predict(X).reshape(-1, 1), np.full(X.shape[0], np.nan).reshape(-1,1)
    except AttributeError:
        return model.predict(X).reshape(-1, 1), np.full(X.shape[0], np.nan).reshape(-1,1)
    
    all_preds = np.stack(preds_per_tree, axis=1)
    yp_mean = np.mean(all_preds, axis=1)
    yp_std = np.std(all_preds, axis=1)
    return (yp_mean.reshape(-1, 1), yp_std.reshape(-1, 1))

def eval_ml_model(model, Xtr, Ytr, Xval, Yval, Xte, Yte, norm=None, clip01=True, quantiles=[0.5]):
    """Evaluate the supplied ML model on train/validation/test data."""
    Xtr2, Xval2, Xte2 = map(flatten_channel_time, (Xtr, Xval, Xte))

    # Fit the model when training data are supplied.
    if (Ytr is not None) and (Ytr.shape[0] > 0):
        if isinstance(model, MLQuantileWrapper):
            model.fit(Xtr2, Ytr)
        elif is_regressor(model):
            model.fit(Xtr2, Ytr.ravel())

    out = {}
    for split, (X2, Y) in {"train": (Xtr2, Ytr), "val": (Xval2, Yval), "test": (Xte2, Yte)}.items():
        if Y is None or Y.shape[0] == 0:
            out[split] = {}
            continue

        # Get predictions.
        if isinstance(model, MLQuantileWrapper):
            yp_n = model.predict(X2).reshape(-1, len(quantiles))
        elif isinstance(model, RandomForestRegressor):
            yp_n = get_rf_prediction_with_std(model, X2)
        elif is_regressor(model):
            yp_n = model.predict(X2).reshape(-1, 1)
        else:
            raise TypeError(f"Unknown ML model type: {type(model)}")

        # Reverse normalization and evaluate.
        if norm is not None:
            Y_den = denorm_y(Y, norm)
            YP_den = denorm_y(yp_n, norm)
            out[split] = summary_metrics(Y_den, YP_den, quantiles=quantiles)
            if isinstance(YP_den, tuple):
                out[split]["y_true_den"] = Y_den
                out[split]["y_pred_den"] = YP_den[0]
            else:
                out[split]["y_true_den"] = Y_den
                out[split]["y_pred_den"] = YP_den
        else:
            out[split] = summary_metrics(Y, yp_n, quantiles=quantiles)
            if isinstance(yp_n, tuple):
                out[split]["y_true_den"] = Y
                out[split]["y_pred_den"] = yp_n[0]
            else:
                out[split]["y_true_den"] = Y
                out[split]["y_pred_den"] = yp_n
        
        if isinstance(yp_n, tuple):
            out[split]["pred"] = yp_n[0]
        else:
            out[split]["pred"] = yp_n
            
    return out

def save_ml(model, path):
    dump(model, path)
