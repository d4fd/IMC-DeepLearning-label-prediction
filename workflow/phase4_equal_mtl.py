"""Equal-weight MTL versus matched mean-pooled image-only heads (seeds 42,43,44).
Frozen per-tile features; train/validation rows only. BCE + unweighted categorical
cross-entropy / train-standardized age Huber; missing auxiliaries masked.
Same optimizer, minibatches, normalization and early stopping as phase4_heads.py.
"""
import submission_paths as _submission_paths
import submission_features as _submission_features
import copy,hashlib,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,roc_auc_score
from phase4_prepare_aux_targets import value_of

R=Path(str(_submission_paths.root())); CONTRACT=R/'phase4_aux_targets_v1.json'
DATA=_submission_paths.dataset() / 'metadata/tissues.parquet'
CACHE=_submission_paths.tokens(); OUT=R/'mtl_outputs'; RESULT=R/'phase4_equal_mtl_results.json'
SEEDS=(42,43,44); EPOCHS=300; PATIENCE=30; BATCH=32; LR=1e-3

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
 return h.hexdigest()

def load(split,c):
 cols=['tissue_id','split','necrose']+c['retained']
 df=pd.read_parquet(DATA,columns=cols,filters=[('split','==',split)])
 assert set(df['split'])=={split}
 df=df[df.necrose.notna()].reset_index(drop=True)
 bags=[]; ids=[]; y=[]; aux={k:[] for k in c['retained']}
 for row in df.itertuples(index=False):
  with np.load(CACHE/split/(row.tissue_id+'.npz')) as z: feature=_submission_features.tile_features(z)
  assert feature.ndim==2 and feature.shape[1]==512 and np.isfinite(feature).all()
  bags.append(feature);ids.append(row.tissue_id);y.append(float(row.necrose))
  for name in aux:
   spec=c['targets'][name]; raw=getattr(row,name)
   if spec['type']=='continuous': val=(float(raw)-spec['train_mean'])/spec['train_std'] if pd.notna(raw) else np.nan
   else:
    code=value_of(name,raw);val=spec['vocabulary'].index(code) if code in spec['vocabulary'] else -1
   aux[name].append(val)
 aux={k:np.asarray(v,dtype=np.float32 if c['targets'][k]['type']=='continuous' else np.int64) for k,v in aux.items()}
 return bags,np.asarray(y,dtype=np.float32),ids,aux

class ImageOnly(nn.Module):
 def __init__(self):
  super().__init__();self.shared=nn.Sequential(nn.Linear(512,128),nn.ReLU())
  self.classifier=nn.Linear(128,1)
 def forward(self,x,mask):
  w=mask.unsqueeze(-1).float();z=(x*w).sum(1)/w.sum(1).clamp(min=1)
  return self.classifier(self.shared(z)).squeeze(-1)

class MTL(nn.Module):
 def __init__(self,c):
  super().__init__();self.shared=nn.Sequential(nn.Linear(512,128),nn.ReLU());self.classifier=nn.Linear(128,1)
  self.aux_heads=nn.ModuleDict({k:nn.Linear(128,1 if c['targets'][k]['type']=='continuous' else len(c['targets'][k]['vocabulary'])) for k in c['retained']})
 def forward(self,x,mask):
  w=mask.unsqueeze(-1).float();z=(x*w).sum(1)/w.sum(1).clamp(min=1)
  z=self.shared(z)
  return self.classifier(z).squeeze(-1),{k:v(z) for k,v in self.aux_heads.items()}

def batches(x,y,aux,rng=None):
 order=np.arange(len(x))
 if rng is not None:rng.shuffle(order)
 for start in range(0,len(x),BATCH):
  ix=order[start:start+BATCH]; n=max(len(x[i]) for i in ix)
  pad=np.zeros((len(ix),n,512),np.float32);mask=np.zeros((len(ix),n),bool)
  for j,i in enumerate(ix): pad[j,:len(x[i])]=x[i];mask[j,:len(x[i])]=True
  yield torch.from_numpy(pad),torch.from_numpy(mask),torch.from_numpy(y[ix]),{k:torch.from_numpy(v[ix]) for k,v in aux.items()}

def validate(model,x,y,aux,device,c):
 model.eval();prob=[]
 with torch.inference_mode():
  for pad,mask,_,_ in batches(x,y,aux):
   result=model(pad.to(device),mask.to(device));logits=result[0] if isinstance(result,tuple) else result
   prob.extend(torch.sigmoid(logits).cpu().tolist())
 p=np.asarray(prob);assert len(p)==len(y)
 coverage={k:{'labeled_cores':int(np.isfinite(v).sum() if c['targets'][k]['type']=='continuous' else (v>=0).sum()),
              'missing_or_unknown_cores':int((~np.isfinite(v)).sum() if c['targets'][k]['type']=='continuous' else (v<0).sum())} for k,v in aux.items()}
 return float(average_precision_score(y,p)),float(roc_auc_score(y,p)),p,coverage

def train_seed(seed,tr,va,mu,sd,c,device,multitask):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 xt=[(v-mu)/sd for v in tr[0]];xv=[(v-mu)/sd for v in va[0]]
 model=(MTL(c) if multitask else ImageOnly()).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=LR)
 rng=np.random.default_rng(seed);best={'epoch':-1,'pr':-1.0};history=[]
 for epoch in range(1,EPOCHS+1):
  model.train();loss_sum={'necrose':0.,**({k:0. for k in c['retained']} if multitask else {})};counts={k:0 for k in loss_sum}
  for x,m,y,targets in batches(xt,tr[1],tr[3],rng):
   x,m,y=x.to(device),m.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
   output=model(x,m);primary,heads=output if multitask else (output,{})
   bce=F.binary_cross_entropy_with_logits(primary,y)
   total=bce;loss_sum['necrose']+=float(bce.detach())*len(y);counts['necrose']+=len(y)
   for name in (c['retained'] if multitask else []):
    label=targets[name].to(device);kind=c['targets'][name]['type']
    present=torch.isfinite(label) if kind=='continuous' else label>=0
    count=int(present.sum())
    if not count:continue
    raw=F.huber_loss(heads[name].squeeze(-1)[present],label[present],delta=1.) if kind=='continuous' else F.cross_entropy(heads[name][present],label[present].long())
    total=total+raw;loss_sum[name]+=float(raw.detach())*count;counts[name]+=count
   if not bool(torch.isfinite(total.detach())):raise ValueError('non-finite loss')
   total.backward();optimizer.step()
  pr,roc,pred,coverage=validate(model,xv,va[1],va[3],device,c)
  raw={k:loss_sum[k]/counts[k] if counts[k] else None for k in counts}
  record={'epoch':epoch,'raw_losses':raw,'effective_weighted_losses':raw.copy(),'label_counts':counts,'val_pr_auc':pr,'val_roc_auc':roc}
  history.append(record)
  if pr>best['pr']:
   best={'pr':pr,'roc':roc,'epoch':epoch,'prediction':pred,'record':record,'coverage':coverage,
         'state':copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})}
  elif epoch-best['epoch']>=PATIENCE:break
 OUT.mkdir(exist_ok=True)
 tag='equal_mtl' if multitask else 'matched_image_only'
 torch.save({'state_dict':best['state'],'seed':seed,'mean':torch.from_numpy(mu),'std':torch.from_numpy(sd),'contract_sha256':sha(CONTRACT)},OUT/f'{tag}_seed{seed}.pt')
 pd.DataFrame({'tissue_id':va[2],'probability':best['prediction']}).to_csv(OUT/f'{tag}_validation_seed{seed}.csv',index=False)
 (OUT/f'{tag}_epoch_log_seed{seed}.json').write_text(json.dumps(history,indent=2)+chr(10))
 return {'seed':seed,'best_epoch':best['epoch'],'epochs_run':epoch,'val_pr_auc':best['pr'],'val_roc_auc':best['roc'],
         'selected_epoch_losses':best['record'],'validation_aux_coverage':best['coverage']}

def main():
 c=json.loads(CONTRACT.read_text());assert len(c['retained'])==9
 assert c['split_artifact_sha256']==sha(R/'splits/rcc2025necrosis_pid_split.csv')
 tr,va=load('train',c),load('validation',c)
 if not len(tr[0]) or not len(va[0]): raise ValueError('Empty development partition')
 z=np.concatenate(tr[0]);mu=z.mean(0).astype(np.float32);sd=(z.std(0)+1e-6).astype(np.float32)
 device='cuda' if torch.cuda.is_available() else 'cpu'
 matched=[]
 for seed in SEEDS:
  baseline=train_seed(seed,tr,va,mu,sd,c,device,multitask=False)
  matched.append(baseline)
  print('MATCHED',seed,baseline['val_pr_auc'],flush=True)
 multitask=[]
 for seed in SEEDS:
  result=train_seed(seed,tr,va,mu,sd,c,device,multitask=True);multitask.append(result)
  print('EQUAL_MTL',seed,result['val_pr_auc'],result['best_epoch'],flush=True)
 base=float(np.mean([v['val_pr_auc'] for v in matched]));mtl=float(np.mean([v['val_pr_auc'] for v in multitask]))
 comparison={'image_only_mean_pr_auc':base,'equal_mtl_mean_pr_auc':mtl,'delta':mtl-base,'threshold':0.01,'retain_equal_mtl':bool(mtl-base>=0.01)}
 report={'protocol':{'cores_train':len(tr[0]),'cores_validation':len(va[0]),'seeds':list(SEEDS),'mean_tiles_over_256_patches':True,
         'normalization':'train tiles only','shared_trainable_layer':'Linear(512,128)+ReLU in BOTH arms','optimizer':'Adam lr=1e-3','batch_cores':BATCH,'max_epochs':EPOCHS,'patience':PATIENCE,
         'primary_loss':'unweighted BCE','categorical_loss':'cross-entropy','continuous_loss':'train-standardized Huber(delta=1)',
         'loss_weights':{'necrose':1.,**{k:1. for k in c['retained']}},'test_used':False,'device':device},
         'aux_manifest_sha256':sha(CONTRACT),'tile_cache_metadata_sha256':sha(CACHE/'cache_metadata.json'),
         'matched_image_only':matched,'equal_mtl':multitask,'comparison':comparison,
         'previous_item3_linear_mean_pr_auc':json.loads((R/'phase4_head_results.json').read_text())['results']['mean']['val_pr_auc_mean']}
 RESULT.write_text(json.dumps(report,indent=2,allow_nan=False)+chr(10))
 print('COMPARISON',comparison,'PHASE4_EQUAL_MTL_OK',flush=True)
if __name__=='__main__':main()
