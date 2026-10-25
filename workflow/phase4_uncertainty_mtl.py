"""Phase 4 item 5: learned uncertainty weighting; train/validation only."""
import copy, hashlib, json, random
import numpy as np, pandas as pd, torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
import phase4_equal_mtl as b
R=b.R; OUT=R/'mtl_outputs'; RESULT=R/'phase4_uncertainty_mtl_results.json'
SEEDS=(42,43,44); BOUNDS=(.01,100.); FLOOR=1e-12
class Model(b.MTL):
 def __init__(self,c):
  super().__init__(c); self.log_vars=nn.ParameterDict({k:nn.Parameter(torch.zeros(())) for k in ['necrose']+c['retained']})
def finite(x): return bool(torch.isfinite(x.detach()).all().item())
def weights(m):
 out={}
 for k,s in m.log_vars.items():
  w=torch.exp(-s)
  if not finite(s) or not finite(w): raise FloatingPointError('non-finite uncertainty '+k)
  v=float(w.detach())
  if not BOUNDS[0]<=v<=BOUNDS[1]: raise ValueError(f'weight collapse {k}: {v}')
  out[k]=v
 return out
def run(seed,tr,va,mu,sd,c,device):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 xt=[(v-mu)/sd for v in tr[0]]; xv=[(v-mu)/sd for v in va[0]]
 m=Model(c).to(device); opt=torch.optim.Adam(m.parameters(),lr=b.LR); rng=np.random.default_rng(seed)
 best={'pr':-1.,'epoch':-1}; history=[]
 for ep in range(1,b.EPOCHS+1):
  m.train(); rs={k:0. for k in m.log_vars}; es={k:0. for k in m.log_vars}; ns={k:0 for k in m.log_vars}
  for x,mask,y,t in b.batches(xt,tr[1],tr[3],rng):
   x,mask,y=x.to(device),mask.to(device),y.to(device); opt.zero_grad(set_to_none=True)
   p,heads=m(x,mask); raw={'necrose':F.binary_cross_entropy_with_logits(p,y)}; counts={'necrose':len(y)}
   for k in c['retained']:
    z=t[k].to(device); cont=c['targets'][k]['type']=='continuous'; ok=torch.isfinite(z) if cont else z>=0; n=int(ok.sum())
    if n:
     raw[k]=F.huber_loss(heads[k].squeeze(-1)[ok],z[ok],delta=1.) if cont else F.cross_entropy(heads[k][ok],z[ok].long()); counts[k]=n
   w=weights(m); eff={}
   for k,l in raw.items():
    if not finite(l): raise FloatingPointError('non-finite raw '+k)
    eff[k]=torch.exp(-m.log_vars[k])*l+m.log_vars[k]
    if not finite(eff[k]): raise FloatingPointError('non-finite effective '+k)
    n=counts[k]; rs[k]+=float(l.detach())*n; es[k]+=float(eff[k].detach())*n; ns[k]+=n
   total=sum(eff.values())
   if not finite(total): raise FloatingPointError('non-finite total')
   total.backward()
   if any(q.grad is not None and not finite(q.grad) for q in m.parameters()): raise FloatingPointError('non-finite gradient')
   opt.step(); weights(m)
  w=weights(m)
  rawavg={k:rs[k]/ns[k] if ns[k] else None for k in ns}; effavg={k:es[k]/ns[k] if ns[k] else None for k in ns}
  if any(v is None or v<=FLOOR for v in rawavg.values()): raise ValueError('raw loss collapse '+str(rawavg))
  m.eval(); pred=[]
  with torch.inference_mode():
   for x,mask,_,_ in b.batches(xv,va[1],va[3]):
    logits,_=m(x.to(device),mask.to(device))
    if not finite(logits): raise FloatingPointError('non-finite validation logits')
    pred.extend(torch.sigmoid(logits).cpu().tolist())
  pred=np.asarray(pred); pr=float(average_precision_score(va[1],pred)); roc=float(roc_auc_score(va[1],pred))
  if len(pred)!=len(va[1]) or not np.isfinite(pred).all() or not np.isfinite([pr,roc]).all(): raise FloatingPointError('invalid predictions/metrics')
  rec={'epoch':ep,'raw_losses':rawavg,'effective_weighted_losses':effavg,'label_counts':ns,'learned_precisions':w,'learned_log_variances':{k:float(v.detach()) for k,v in m.log_vars.items()},'val_pr_auc':pr,'val_roc_auc':roc}; history.append(rec)
  if pr>best['pr']: best={'pr':pr,'roc':roc,'epoch':ep,'pred':pred.copy(),'record':rec,'state':copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()})}
  elif ep-best['epoch']>=b.PATIENCE: break
 OUT.mkdir(exist_ok=True); torch.save({'state_dict':best['state'],'seed':seed,'mean':torch.from_numpy(mu),'std':torch.from_numpy(sd)},OUT/f'uncertainty_mtl_seed{seed}.pt')
 pd.DataFrame({'tissue_id':va[2],'probability':best['pred']}).to_csv(OUT/f'uncertainty_mtl_validation_seed{seed}.csv',index=False)
 (OUT/f'uncertainty_mtl_epoch_log_seed{seed}.json').write_text(json.dumps(history,indent=2,allow_nan=False)+'\n')
 coverage=b.validate(m,xv,va[1],va[3],device,c)[3]
 return {'seed':seed,'best_epoch':best['epoch'],'epochs_run':ep,'val_pr_auc':best['pr'],'val_roc_auc':best['roc'],'selected_epoch_losses':best['record'],'validation_aux_coverage':coverage,'learned_weights_precision':best['record']['learned_precisions'],'learned_log_variances':best['record']['learned_log_variances']}
def main():
 c=json.loads(b.CONTRACT.read_text()); assert len(c['retained'])==9
 tr=b.load('train',c); va=b.load('validation',c)
 if not len(tr[0]) or not len(va[0]): raise ValueError('Empty development partition')
 z=np.concatenate(tr[0]); mu=z.mean(0).astype(np.float32); sd=(z.std(0)+1e-6).astype(np.float32)
 device='cuda' if torch.cuda.is_available() else 'cpu'; runs=[]; failure=None
 try:
  for seed in SEEDS:
   q=run(seed,tr,va,mu,sd,c,device); runs.append(q); print('UNCERTAINTY_MTL',seed,q['val_pr_auc'],q['best_epoch'],flush=True)
 except Exception as e: failure={'type':type(e).__name__,'message':str(e)}
 prior=json.loads(b.RESULT.read_text()); mean=float(np.mean([q['val_pr_auc'] for q in runs])) if len(runs)==3 else None
 img=float(prior['comparison']['image_only_mean_pr_auc']); equal=float(prior['comparison']['equal_mtl_mean_pr_auc']); delta=mean-img if mean is not None else None
 report={'protocol':{'seeds':list(SEEDS),'train_cores':len(tr[0]),'validation_cores':len(va[0]),'features':'frozen cached tokens, mean of 256 patch summaries','normalization':'train tiles only','pooling':'mean tiles per core','shared_layer':'Linear(512,128)+ReLU','loss':'per present task exp(-s_t)*L_t+s_t; s_t initialized zero','raw_losses':{'primary':'BCEWithLogits','categorical':'cross-entropy','continuous':'train-standardized Huber(delta=1)'},'optimizer':'Adam lr=1e-3','batch_cores':32,'max_epochs':300,'early_stopping':'validation PR-AUC, patience 30','collapse_gates':{'raw_loss':'reject <=1e-12','precision':'reject exp(-s_t) outside [0.01,100] at any step','non_finite':'reject any loss, uncertainty, gradient, prediction, or metric'},'test_used':False,'encoder_trained':False,'device':device},'aux_manifest_sha256':hashlib.sha256(b.CONTRACT.read_bytes()).hexdigest(),'tile_cache_metadata_sha256':b.sha(b.CACHE/'cache_metadata.json'),'matched_image_only_mean_validation_pr_auc':img,'equal_weight_mtl_mean_validation_pr_auc':equal,'uncertainty_mtl':runs,'uncertainty_mtl_mean_validation_pr_auc':mean,'delta_vs_image_only':delta,'delta_vs_equal_weight_mtl':mean-equal if mean is not None else None,'retention_threshold':0.01,'retained':failure is None and mean is not None and delta>=.01,'failure':failure}
 RESULT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); print('REPORT',json.dumps({k:report[k] for k in ['uncertainty_mtl_mean_validation_pr_auc','delta_vs_image_only','delta_vs_equal_weight_mtl','retained','failure']}),flush=True)
if __name__=='__main__': main()
