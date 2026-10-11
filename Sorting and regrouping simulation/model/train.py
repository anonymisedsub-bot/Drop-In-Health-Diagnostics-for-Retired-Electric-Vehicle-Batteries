"""Train the configured t46_i2_s0 round with an explicit epoch budget."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch
import config as cfg
from main import train_one_round

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from portable_io import validate_output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent/'results/runs/training')
    args=p.parse_args()
    try:
        args.output=validate_output(args.output)
    except ValueError as exc:
        p.error(str(exc))
    here=Path(__file__).resolve().parent
    torch.set_num_threads(2)
    cfg.device='cpu'; cfg.VSEG_LOW=2.73; cfg.VSEG_UPP=3.03
    cfg.num_finetune_epochs=args.epochs; cfg.patience_epochs=args.epochs
    splits=json.loads((here.parent/'results/model/splits_init.json').read_text())
    train=np.array(json.loads((here.parent/'results/model/train_indices.json').read_text()))
    pool=np.setdiff1d(splits['pool'],train)
    args.output.mkdir(parents=True,exist_ok=True)
    train_one_round(train,pool,np.array(splits['val']),np.array(splits['test']),np.load(here.parent/'results/model/V_init.npy'),0,args.output)
    (args.output/'training_scope.json').write_text(json.dumps(dict(epochs=args.epochs,scope='new short training; not historical checkpoint reproduction',condition='t46_i2_s0',device='cpu'),indent=2))
