#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import accuracy_score,f1_score
from sklearn.svm import SVC
from experiments.sipakmed.paper_baselines_v1.ga_cnn import BinaryGA,GoogLeNetResNet18FeatureExtractor,extract_or_load_features
from experiments.sipakmed.paper_baselines_v1.train_mtfm_sipakmed import DEFAULT_MANIFEST,DEFAULT_RESULTS,EXPECTED_CLASS_NAMES,classification_metrics,make_internal_split,validate_manifest
ROOT=Path(__file__).resolve().parents[3]; NORMAL=(0,1,4)

def atomic_json(p,x):
    t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n"); t.replace(p)

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def metrics(y,p):
    out=classification_metrics(y,p); bt=np.asarray([0 if int(v) in NORMAL else 1 for v in y]); bp=np.asarray([0 if int(v) in NORMAL else 1 for v in p])
    out["binary_accuracy"]=float(accuracy_score(bt,bp)); out["binary_macro_f1"]=float(f1_score(bt,bp,average="macro",zero_division=0)); return out

def fold_run(fold,manifest,features,idx,root,args):
    s=args.seed+fold; seed(s); fr=root/f"fold_{fold}"; fr.mkdir()
    outer=manifest.loc[manifest.fold!=fold].copy(); held=manifest.loc[manifest.fold==fold].copy(); fit,select=make_internal_split(outer,s); frames={"fit":fit,"select":select,"heldout":held}; groups={n:set(f.group_id.astype(str)) for n,f in frames.items()}; inter={f"{a}_x_{b}":len(groups[a]&groups[b]) for a,b in (("fit","select"),("fit","heldout"),("select","heldout"))}
    if any(inter.values()): raise RuntimeError(f"group leakage {inter}")
    sd=fr/"split"; sd.mkdir()
    for n,f in frames.items(): f.to_csv(sd/f"{n}.csv",index=False)
    atomic_json(fr/"split_audit.json",{"fold":fold,"counts":{n:len(f) for n,f in frames.items()},"groups":{n:len(g) for n,g in groups.items()},"pairwise_group_intersections":inter,"classes":EXPECTED_CLASS_NAMES,"fit_views":1})
    atomic_json(root/"status.json",{"state":"training","method":"GA-GoogLeNet-ResNet18-RBF-SVM","fold":fold,"phase":"ga","updated_utc":datetime.now(timezone.utc).isoformat()})
    xfit=np.asarray(features[[idx[p] for p in fit.image_path]],dtype=np.float32); xsel=np.asarray(features[[idx[p] for p in select.image_path]],dtype=np.float32); xheld=np.asarray(features[[idx[p] for p in held.image_path]],dtype=np.float32)
    ga=BinaryGA(args.population,args.generations,args.mutation_count,s,args.svm_c); chosen,history=ga.optimize(xfit,fit.label.to_numpy(dtype=np.int64),args.ga_inner_fraction)
    np.save(fr/"selected_feature_indices.npy",chosen); (fr/"ga_history.json").write_text(json.dumps(history,indent=2)+"\n")
    clf=SVC(kernel="rbf",gamma="scale",C=args.svm_c,cache_size=args.svm_cache_mb); clf.fit(xfit[:,chosen],fit.label.to_numpy(dtype=np.int64))
    ps=clf.predict(xsel[:,chosen]); ph=clf.predict(xheld[:,chosen]); sm=metrics(select.label.to_numpy(),ps); hm=metrics(held.label.to_numpy(),ph)
    pd.DataFrame({"image_path":held.image_path.to_list(),"label":held.label.to_list(),"prediction":ph.tolist()}).to_csv(fr/"heldout_predictions.csv",index=False)
    result={"method":"GA-GoogLeNet-ResNet18-RBF-SVM","dataset":"SIPaKMeD","fold":fold,"seed":s,"n_fit":len(fit),"n_select":len(select),"n_heldout":len(held),"feature_dim_raw":int(features.shape[1]),"ga_population":args.population,"ga_generations":args.generations,"mutation_count":args.mutation_count,"svm_kernel":"rbf","svm_c":args.svm_c,"selected_feature_count":int(chosen.size),"selection_metrics":sm,"heldout_metrics":hm,"heldout_evaluation_count":1,"class_names":EXPECTED_CLASS_NAMES,"predictions_recorded":True}
    atomic_json(fr/"result.json",result); atomic_json(root/"status.json",{"state":"fold_complete","method":result["method"],"fold":fold,"result":hm,"updated_utc":datetime.now(timezone.utc).isoformat()}); print(f"FOLD COMPLETE fold={fold} accuracy={hm['accuracy']:.4f} macro_f1={hm['macro_f1']:.4f} selected={chosen.size}",flush=True); return result

def summary(root,results):
    rows=[{"fold":r["fold"],**r["heldout_metrics"]} for r in results]; t=pd.DataFrame(rows).sort_values("fold"); t.to_csv(root/"fold_metrics.csv",index=False); names=("accuracy","macro_f1","macro_precision","macro_recall","binary_accuracy","binary_macro_f1"); out={"method":"GA-GoogLeNet-ResNet18-RBF-SVM","dataset":"SIPaKMeD","completed_folds":len(results),"metrics_mean":{n:float(t[n].mean()) for n in names},"metrics_sample_sd":{n:float(t[n].std(ddof=1)) for n in names},"folds":rows}; atomic_json(root/"summary.json",out); return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,default=DEFAULT_MANIFEST); p.add_argument("--results-root",type=Path,default=DEFAULT_RESULTS); p.add_argument("--run-name",required=True); p.add_argument("--cache-root",type=Path,default=ROOT/"cache/sipakmed_ga_googlenet_resnet18"); p.add_argument("--device",default="cuda:0"); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--workers",type=int,default=0); p.add_argument("--population",type=int,default=100); p.add_argument("--generations",type=int,default=50); p.add_argument("--mutation-count",type=int,default=6); p.add_argument("--ga-inner-fraction",type=float,default=.25); p.add_argument("--svm-c",type=float,default=5000.0); p.add_argument("--svm-cache-mb",type=int,default=4096); p.add_argument("--seed",type=int,default=42); p.add_argument("--smoke",action="store_true"); a=p.parse_args()
    dev=torch.device(a.device if a.device.startswith("cuda") and torch.cuda.is_available() else "cpu"); root=a.results_root/a.run_name
    if a.smoke:
        root.mkdir(parents=True,exist_ok=False); m=GoogLeNetResNet18FeatureExtractor().to(dev).eval()
        with torch.no_grad(): out=m(torch.zeros(2,3,224,224,device=dev))
        x=np.random.default_rng(a.seed).normal(size=(40,12)).astype(np.float32); y=np.tile(np.arange(5),8); chosen,h=BinaryGA(4,2,2,a.seed,10).optimize(x,y); atomic_json(root/"smoke.json",{"state":"passed","input_batch":[2,3,224,224],"feature_shape":list(out.shape),"selected_feature_count":int(chosen.size),"ga_generations":len(h)}); print(json.dumps({"state":"passed","feature_shape":list(out.shape),"selected_feature_count":int(chosen.size)})); return
    manifest=validate_manifest(a.manifest)
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True); method="GA-GoogLeNet-ResNet18-RBF-SVM"; src={"method":method,"paper_title":"Deep Features Selection through Genetic Algorithm for Cervical Pre-cancerous Cell Classification","paper_doi":"10.1007/s11042-022-13736-9","official_code":"https://github.com/Rohit-Kundu/Cervical-Cancer-CNN-GA","architecture":["ImageNet pretrained GoogLeNet 1024-d features","ImageNet pretrained ResNet-18 512-d features","feature concatenation","binary Genetic Algorithm feature selection","RBF SVM with C=5000"],"adaptations":["verified SIPaKMeD 4049-image group-disjoint five-fold manifest","GA fitness uses only an inner split of the outer-train data","held-out fold is evaluated once","official feature extractor resize+ToTensor preprocessing is retained"],"config":{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}}; atomic_json(root/"source_fidelity.json",src); atomic_json(root/"status.json",{"state":"feature_extraction","method":method,"updated_utc":datetime.now(timezone.utc).isoformat()})
    paths=manifest.image_path.astype(str).tolist(); feat=extract_or_load_features(paths,a.cache_root,dev,a.batch_size,a.workers); idx={p:i for i,p in enumerate(paths)}; results=[]
    try:
        for fold in range(5): results.append(fold_run(fold,manifest,feat,idx,root,a))
        out=summary(root,results); atomic_json(root/"status.json",{"state":"completed","method":method,"completed_folds":5,"summary":out,"updated_utc":datetime.now(timezone.utc).isoformat()}); print(json.dumps(out,ensure_ascii=False))
    except Exception as e:
        atomic_json(root/"status.json",{"state":"failed","method":method,"error":repr(e),"updated_utc":datetime.now(timezone.utc).isoformat()}); raise

if __name__=="__main__": main()
