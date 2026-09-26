#!/usr/bin/env python3
"""Grouped five-fold reproduction of PCA+GWO deep-feature selection on SIPaKMeD."""

from __future__ import annotations
import argparse,json,random,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
from sklearn.svm import SVC
from experiments.sipakmed.paper_baselines_v1.pca_gwo import BinaryGWO,extract_or_load_features,fit_pca_99,transform_pca
from experiments.sipakmed.paper_baselines_v1.train_mtfm_sipakmed import (
    DEFAULT_MANIFEST,DEFAULT_RESULTS,EXPECTED_CLASS_NAMES,classification_metrics,make_internal_split,validate_manifest,
)

ROOT=Path(__file__).resolve().parents[3]
BINARY_NORMAL_LABELS=(0,1,4)

def atomic_json(path,payload):
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); tmp.replace(path)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def binary_metrics(y_true,y_pred):
    bt=np.asarray([0 if int(y) in BINARY_NORMAL_LABELS else 1 for y in y_true])
    bp=np.asarray([0 if int(y) in BINARY_NORMAL_LABELS else 1 for y in y_pred])
    return {"binary_accuracy":float(accuracy_score(bt,bp)),"binary_macro_f1":float(f1_score(bt,bp,average="macro",zero_division=0))}

def metrics(y_true,y_pred):
    out=classification_metrics(y_true,y_pred); out.update(binary_metrics(y_true,y_pred)); return out

def save_pca_basis(fold_root,pca,k):
    np.save(fold_root/"pca_components.npy",np.asarray(pca.components_[:k],dtype=np.float32))
    np.save(fold_root/"pca_mean.npy",np.asarray(pca.mean_,dtype=np.float32))

def train_fold(fold,manifest,feature_table,index_by_path,run_root,args):
    fold_seed=args.seed+fold; seed_everything(fold_seed); fold_root=run_root/f"fold_{fold}"; fold_root.mkdir(parents=False,exist_ok=False)
    outer_train=manifest.loc[manifest.fold!=fold].copy(); heldout=manifest.loc[manifest.fold==fold].copy()
    fit,select=make_internal_split(outer_train,fold_seed); frames={"fit":fit,"select":select,"heldout":heldout}
    groups={n:set(f.group_id.astype(str)) for n,f in frames.items()}
    intersections={f"{a}_x_{b}":len(groups[a]&groups[b]) for a,b in (("fit","select"),("fit","heldout"),("select","heldout"))}
    if any(intersections.values()): raise RuntimeError(f"fold {fold} group leakage: {intersections}")
    split_dir=fold_root/"split"; split_dir.mkdir()
    for name,frame in frames.items(): frame.to_csv(split_dir/f"{name}.csv",index=False)
    atomic_json(fold_root/"split_audit.json",{"fold":fold,"counts":{n:len(f) for n,f in frames.items()},"groups":{n:len(v) for n,v in groups.items()},"pairwise_group_intersections":intersections,"classes":EXPECTED_CLASS_NAMES,"fit_views":1})
    atomic_json(run_root/"status.json",{"state":"training","method":"PCA-GWO-ResNet50-VGG16-RBF-SVM","fold":fold,"phase":"pca","updated_utc":datetime.now(timezone.utc).isoformat()})
    fit_x=np.asarray(feature_table[[index_by_path[p] for p in fit.image_path]],dtype=np.float32)
    select_x=np.asarray(feature_table[[index_by_path[p] for p in select.image_path]],dtype=np.float32)
    heldout_x=np.asarray(feature_table[[index_by_path[p] for p in heldout.image_path]],dtype=np.float32)
    pca,k,retained=fit_pca_99(fit_x,variance_target=args.variance_target,max_components=args.pca_max_components,seed=fold_seed)
    z_fit=transform_pca(pca,k,fit_x); z_select=transform_pca(pca,k,select_x); z_heldout=transform_pca(pca,k,heldout_x)
    save_pca_basis(fold_root,pca,k)
    np.save(fold_root/"pca_explained_variance_ratio.npy",np.asarray(pca.explained_variance_ratio_[:k],dtype=np.float32))
    del fit_x,select_x,heldout_x
    atomic_json(run_root/"status.json",{"state":"training","method":"PCA-GWO-ResNet50-VGG16-RBF-SVM","fold":fold,"phase":"gwo","pca_components":k,"pca_retained_variance":retained,"updated_utc":datetime.now(timezone.utc).isoformat()})
    gwo=BinaryGWO(args.population,args.iterations,fold_seed); selected,history=gwo.optimize(z_fit,fit.label.to_numpy(dtype=np.int64),inner_fraction=args.gwo_inner_fraction)
    if selected.size==0: raise RuntimeError("GWO returned no features")
    np.save(fold_root/"selected_feature_indices.npy",selected)
    (fold_root/"gwo_history.json").write_text(json.dumps(history,indent=2)+"\n",encoding="utf-8")
    classifier=SVC(kernel="rbf",C=args.svm_c,gamma=args.svm_gamma,cache_size=args.svm_cache_mb)
    classifier.fit(z_fit[:,selected],fit.label.to_numpy(dtype=np.int64))
    select_pred=classifier.predict(z_select[:,selected]); select_metrics=metrics(select.label.to_numpy(),select_pred)
    heldout_pred=classifier.predict(z_heldout[:,selected]); heldout_metrics=metrics(heldout.label.to_numpy(),heldout_pred)
    pd.DataFrame({"image_path":heldout.image_path.to_list(),"label":heldout.label.to_list(),"prediction":heldout_pred.tolist()}).to_csv(fold_root/"heldout_predictions.csv",index=False)
    result={"method":"PCA-GWO-ResNet50-VGG16-RBF-SVM","dataset":"SIPaKMeD","fold":fold,"seed":fold_seed,"n_fit":len(fit),"n_select":len(select),"n_heldout":len(heldout),"feature_dim_raw":int(feature_table.shape[1]),"pca_components":int(k),"pca_variance_target":args.variance_target,"pca_retained_variance":retained,"gwo_population":args.population,"gwo_iterations":args.iterations,"selected_feature_count":int(selected.size),"svm_kernel":"rbf","selection_metrics":select_metrics,"heldout_metrics":heldout_metrics,"heldout_evaluation_count":1,"class_names":EXPECTED_CLASS_NAMES,"predictions_recorded":True}
    atomic_json(fold_root/"result.json",result); atomic_json(run_root/"status.json",{"state":"fold_complete","method":result["method"],"fold":fold,"result":heldout_metrics,"updated_utc":datetime.now(timezone.utc).isoformat()})
    print(f"FOLD COMPLETE fold={fold} accuracy={heldout_metrics['accuracy']:.4f} macro_f1={heldout_metrics['macro_f1']:.4f} selected={selected.size}",flush=True)
    return result

def write_summary(run_root,results):
    rows=[{"fold":r["fold"],**r["heldout_metrics"]} for r in results]; table=pd.DataFrame(rows).sort_values("fold"); table.to_csv(run_root/"fold_metrics.csv",index=False)
    names=("accuracy","macro_f1","macro_precision","macro_recall","binary_accuracy","binary_macro_f1")
    summary={"method":"PCA-GWO-ResNet50-VGG16-RBF-SVM","dataset":"SIPaKMeD","completed_folds":len(results),"metrics_mean":{n:float(table[n].mean()) for n in names},"metrics_sample_sd":{n:float(table[n].std(ddof=1)) for n in names},"folds":rows}
    atomic_json(run_root/"summary.json",summary); return summary

def run_smoke(device,args,root):
    from experiments.sipakmed.paper_baselines_v1.pca_gwo import ResNet50VGG16FeatureExtractor
    model=ResNet50VGG16FeatureExtractor().to(device).eval()
    with torch.no_grad(): out=model(torch.zeros(2,3,224,224,device=device))
    x=np.random.default_rng(args.seed).normal(size=(40,12)).astype(np.float32); y=np.tile(np.arange(5),8)
    selected,history=BinaryGWO(4,2,args.seed).optimize(x,y)
    atomic_json(root/"smoke.json",{"state":"passed","input_batch":[2,3,224,224],"feature_shape":list(out.shape),"selected_feature_count":int(selected.size),"gwo_iterations":len(history),"feature_extractors":["ResNet-50","VGG-16"]})
    print(json.dumps({"state":"passed","feature_shape":list(out.shape),"selected_feature_count":int(selected.size)}),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,default=DEFAULT_MANIFEST); parser.add_argument("--results-root",type=Path,default=DEFAULT_RESULTS)
    parser.add_argument("--run-name",required=True); parser.add_argument("--cache-root",type=Path,default=ROOT/"cache/sipakmed_pca_gwo_resnet50_vgg16")
    parser.add_argument("--device",default="cuda:0"); parser.add_argument("--batch-size",type=int,default=8); parser.add_argument("--workers",type=int,default=0)
    parser.add_argument("--population",type=int,default=20); parser.add_argument("--iterations",type=int,default=20); parser.add_argument("--gwo-inner-fraction",type=float,default=.25)
    parser.add_argument("--variance-target",type=float,default=.99); parser.add_argument("--pca-max-components",type=int,default=1024)
    parser.add_argument("--svm-c",type=float,default=1.0); parser.add_argument("--svm-gamma",default="scale"); parser.add_argument("--svm-cache-mb",type=int,default=4096); parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--smoke",action="store_true"); args=parser.parse_args()
    device=torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if args.smoke:
        root=args.results_root/args.run_name; root.mkdir(parents=True,exist_ok=False); run_smoke(device,args,root); return
    manifest=validate_manifest(args.manifest); run_root=args.results_root/args.run_name
    if run_root.exists(): raise FileExistsError(f"run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    source={"method":"PCA-GWO-ResNet50-VGG16-RBF-SVM","paper_title":"Cervical Cytology Classification Using PCA and GWO Enhanced Deep Features Selection","paper_doi":"10.1007/s42979-021-00741-2","paper_arxiv":"2106.04919","official_code":"https://github.com/DVLP-CMATERJU/Two-Step-Feature-Enhancement","paper_reported_sipakmed":{"accuracy":.9787,"precision":.9856,"recall":.9912,"f1":.9889},"architecture":["ImageNet pretrained ResNet-50 conv feature map","ImageNet pretrained VGG-16 conv feature map","feature concatenation","PCA variance target 99%","binary Grey Wolf Optimizer feature selection","RBF SVM"],"adaptations":["verified SIPaKMeD 4049-image group-disjoint five-fold manifest","PCA and GWO fit only on outer-train data; inner selection split is isolated","held-out fold is evaluated once","randomized PCA is used with a configurable 1024-component cap for tractable five-fold reproduction","official KNN fitness and RBF-SVM final classifier are retained"],"config":{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}}
    atomic_json(run_root/"source_fidelity.json",source); atomic_json(run_root/"status.json",{"state":"feature_extraction","method":source["method"],"updated_utc":datetime.now(timezone.utc).isoformat()})
    paths=manifest.image_path.astype(str).tolist(); table=extract_or_load_features(paths,args.cache_root,device,args.batch_size,args.workers); index_by_path={p:i for i,p in enumerate(paths)}
    results=[]
    try:
        for fold in range(5):
            result_path=run_root/f"fold_{fold}/result.json"
            if result_path.is_file(): results.append(json.loads(result_path.read_text(encoding="utf-8"))); continue
            results.append(train_fold(fold,manifest,table,index_by_path,run_root,args))
        summary=write_summary(run_root,results); atomic_json(run_root/"status.json",{"state":"completed","method":source["method"],"completed_folds":5,"summary":summary,"updated_utc":datetime.now(timezone.utc).isoformat()})
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    except Exception as exc:
        atomic_json(run_root/"status.json",{"state":"failed","method":source["method"],"error":repr(exc),"updated_utc":datetime.now(timezone.utc).isoformat()}); raise

if __name__=="__main__": main()
