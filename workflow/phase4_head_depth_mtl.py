"""Phase 4 item 7 deeper-head comparison on frozen train/validation tokens only."""
import json
from pathlib import Path
from torch import nn
import phase4_equal_mtl as baseline

class DeeperMTL(nn.Module):
 def __init__(self, contract):
  super().__init__()
  self.shared=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
  self.classifier=nn.Linear(64,1)
  self.aux_heads=nn.ModuleDict({name:nn.Linear(64,1 if contract['targets'][name]['type']=='continuous' else len(contract['targets'][name]['vocabulary'])) for name in contract['retained']})
 def forward(self,x,mask):
  w=mask.unsqueeze(-1).float();z=(x*w).sum(1)/w.sum(1).clamp(min=1);z=self.shared(z)
  return self.classifier(z).squeeze(-1),{name:head(z) for name,head in self.aux_heads.items()}

def main():
 original=baseline.RESULT
 shallow=json.loads(original.read_text())
 out=baseline.R/'phase4_head_depth_mtl_results.json'
 baseline.MTL=DeeperMTL
 baseline.RESULT=out
 # Write to a dedicated directory so the deeper run cannot overwrite the
 # shallow all-nine-auxiliary equal-weight MTL artifacts in mtl_outputs/.
 baseline.OUT=baseline.R/'mtl_outputs'/'head_depth'
 baseline.main()
 deeper=json.loads(out.read_text())
 deeper['protocol']['shared_trainable_layer']='Linear(512,128)+ReLU -> Linear(128,64)+ReLU (deeper arm)'
 shallow_runs=shallow['equal_mtl']
 deeper_runs=deeper.pop('equal_mtl')
 shallow_mean=float(sum(r['val_pr_auc'] for r in shallow_runs)/len(shallow_runs))
 deeper_mean=float(sum(r['val_pr_auc'] for r in deeper_runs)/len(deeper_runs))
 deeper['item7_protocol']={
  'only_change':'added Linear(128,64)+ReLU after the retained Linear(512,128)+ReLU shared layer',
  'test_used':False,'selected_on_higher_mean_pr_auc':True,'tie_rule':'retain shallow'}
 deeper['shallow']=shallow_runs
 deeper['deeper']=deeper_runs
 deeper['comparison']={'shallow_mean_pr_auc':shallow_mean,'deeper_mean_pr_auc':deeper_mean,'delta_deeper_minus_shallow':deeper_mean-shallow_mean,'selected_shared_head':'Linear(512,128)+ReLU -> Linear(128,64)+ReLU' if deeper_mean>shallow_mean else 'Linear(512,128)+ReLU'}
 out.write_text(json.dumps(deeper,indent=2,allow_nan=False)+'\n')
 print('PHASE4_HEAD_DEPTH_MTL_OK',deeper['comparison'],flush=True)
if __name__=='__main__':main()
