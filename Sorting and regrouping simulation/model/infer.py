"""Authentic historical t46_i2_s0 checkpoint inference; no training or label fitting on test."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as cfg
from data import load_all_mats, build_ordered_windows, build_subset, fit_normalizer_tail, fit_normalizer_cap, inverse_cap
from main import _apply_norm_to_pack, apply_segment_availability_to_pack
from model import ConditionalDecoder

def assert_reference_identity(result, reference):
    """Require identical cell ordering and measured labels before comparing values."""
    np.testing.assert_array_equal(result.cell_id.to_numpy(),reference.cell_id.to_numpy(),err_msg='Reference cell identity/order mismatch')
    np.testing.assert_allclose(result.y_true.to_numpy(),reference.y_true.to_numpy(),rtol=0,atol=5e-7,err_msg='Reference measured-label mismatch')


def run(output):
    torch.set_num_threads(2)
    cfg.device = 'cpu'
    cfg.DATA_MAT_PATH = str(HERE.parent/'inputs/LFP_data_processed.mat')
    cfg.VSEG_LOW = 2.5 + 46*0.005
    cfg.VSEG_UPP = cfg.VSEG_LOW + 0.3
    V,Q = load_all_mats(cfg.DATA_MAT_PATH)
    split = json.loads((HERE.parent/'results/model/splits_init.json').read_text())
    train = np.array(json.loads((HERE.parent/'results/model/train_indices.json').read_text()))
    test = np.array(split['test'])
    assert not set(train) & set(test)
    windows,hi,grid = build_ordered_windows(cfg)
    source = build_subset(V,Q,train,cfg,windows,hi)
    tn = fit_normalizer_tail(source['X_tail'],cfg.normalization_mode)
    cn = fit_normalizer_cap(source['y_cap'])
    pack = _apply_norm_to_pack(build_subset(V,Q,test,cfg,windows,hi),tn,cn)
    pack = apply_segment_availability_to_pack(pack,test,np.load(HERE.parent/'results/model/V_init.npy'),windows,grid,'test')
    model = ConditionalDecoder(cfg)
    model.load_state_dict(torch.load(HERE.parent/'results/model/best.pt',map_location='cpu',weights_only=True))
    model.eval()
    predictions=[]
    with torch.inference_mode():
        for start in range(0,len(test),100):
            segments=[]; owners=[]
            for i,segs in enumerate(pack['S_list'][start:start+100]):
                segments.extend(segs); owners.extend([i]*len(segs))
            s=torch.tensor(np.stack(segments)) if segments else torch.empty(0,2,cfg.SEG_LEN)
            _,cap,_=model(torch.tensor(pack['X_tail'][start:start+100]),s,torch.tensor(owners,dtype=torch.long),torch.tensor(pack['z_meta'][start:start+100]))
            predictions.append(cap.numpy())
    values=inverse_cap(np.concatenate(predictions),cn)
    pred=values[:,1]; sigma=(values[:,-1]-values[:,0])/(norm.ppf(.9)-norm.ppf(.1)+1e-12)
    result=pd.DataFrame(dict(cell_id=test,y_true=Q[-1,test],y_pred=pred,raw_lower=pred-1.96*sigma,raw_upper=pred+1.96*sigma))
    reference=pd.read_csv(HERE.parent/'inputs/prepared/t46_i2_s0.csv')
    assert_reference_identity(result,reference)
    errors={col:float(np.max(np.abs(result[col]-reference[col]))) for col in ['y_pred','raw_lower','raw_upper']}
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    result.to_csv(output/'model_predictions_t46_i2_s0.csv',index=False)
    boundaries={name:int(np.count_nonzero(a!=b)) for name,a,b in [('point14',result.y_pred>=14,reference.y_pred>=14),('conservative14_3',result.y_pred>14.3,reference.y_pred>14.3),('lower14',result.raw_lower>=14,reference.lower>=14)]}
    report=dict(condition='t46_i2_s0',n_test=len(test),n_training=len(train),device='cpu',max_absolute_error_ah=errors,tolerance_ah=1e-3,tolerance_type='empirical reproduction tolerance; residual cause not established',boundary_disagreements=boundaries,passed=max(errors.values())<1e-3,physical_pack_validation=False)
    (output/'model_inference_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report['passed']: raise AssertionError('Checkpoint predictions differ from bundled historical workbook; investigate configuration.')

if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv)>1 else HERE.parent/'results/runs/model')
