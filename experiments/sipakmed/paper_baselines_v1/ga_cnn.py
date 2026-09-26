#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from torchvision.models import GoogLeNet_Weights,ResNet18_Weights,googlenet,resnet18
from experiments.sipakmed.paper_baselines_v1.train_mtfm_sipakmed import IMAGE_SIZE

MEAN=np.zeros(3,dtype=np.float32); STD=np.ones(3,dtype=np.float32)

def raw_image_tensor(path):
    from PIL import Image
    with Image.open(path) as source: image=source.convert("RGB").resize((IMAGE_SIZE,IMAGE_SIZE),Image.Resampling.BILINEAR)
    return torch.from_numpy(np.ascontiguousarray(np.asarray(image,dtype=np.float32).transpose(2,0,1)/255.0)).float()

class RawFeatureDataset(torch.utils.data.Dataset):
    def __init__(self,paths): self.paths=paths
    def __len__(self): return len(self.paths)
    def __getitem__(self,i): return raw_image_tensor(self.paths[i]),i

class GoogLeNetResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        g=googlenet(weights=GoogLeNet_Weights.DEFAULT,aux_logits=True); r=resnet18(weights=ResNet18_Weights.DEFAULT)
        g.fc=nn.Identity(); r.fc=nn.Identity(); self.g=g; self.r=r
        for p in self.parameters(): p.requires_grad_(False)
    @torch.inference_mode()
    def forward(self,x):
        return torch.cat((self.g(x),self.r(x)),1)

def _write_json(path,payload):
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(payload,indent=2)+"\n"); tmp.replace(path)

def _signature(paths):
    h=hashlib.sha256()
    for p in paths: h.update(p.encode()); h.update(b"\n")
    return h.hexdigest()

def extract_or_load_features(paths,cache_root,device,batch_size=8,workers=0):
    cache_root.mkdir(parents=True,exist_ok=True); fp=cache_root/"googlenet_resnet18_features.npy"; mp=cache_root/"metadata.json"; sig=_signature(paths)
    if fp.is_file() and mp.is_file():
        m=json.loads(mp.read_text())
        if m.get("path_signature")==sig and m.get("count")==len(paths): return np.load(fp,mmap_mode="r")
    loader=torch.utils.data.DataLoader(RawFeatureDataset(paths),batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=device.type=="cuda",persistent_workers=workers>0)
    model=GoogLeNetResNet18FeatureExtractor().to(device).eval(); out=None; tmp=cache_root/"googlenet_resnet18_features.tmp.npy"
    if tmp.exists(): tmp.unlink()
    for images,indices in loader:
        f=model(images.to(device,non_blocking=True)).float().cpu().numpy()
        if out is None: out=np.lib.format.open_memmap(tmp,mode="w+",dtype=np.float32,shape=(len(paths),int(f.shape[1])))
        out[indices.numpy()]=f
    if out is None: raise RuntimeError("no images")
    out.flush(); dim=int(f.shape[1]); del out,model
    if device.type=="cuda": torch.cuda.empty_cache()
    tmp.replace(fp); _write_json(mp,{"method":"GA-GoogLeNet-ResNet18-RBF-SVM","feature_extractors":["GoogLeNet","ResNet-18"],"feature_dim":dim,"count":len(paths),"path_signature":sig,"weights":"ImageNet pretrained torchvision weights","input_transform":"resize 224 and ToTensor only, matching official extractor"})
    return np.load(fp,mmap_mode="r")

class BinaryGA:
    def __init__(self,population=100,generations=50,mutation_count=6,seed=42,svm_c=5000.0):
        if population<4 or generations<1: raise ValueError("invalid GA configuration")
        self.population=int(population); self.generations=int(generations); self.mutation_count=int(mutation_count); self.rng=np.random.default_rng(seed); self.svm_c=float(svm_c)
    def optimize(self,x,y,inner_fraction=.25):
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.svm import SVC
        split=StratifiedShuffleSplit(n_splits=1,test_size=inner_fraction,random_state=int(self.rng.integers(0,2**31-1)))
        tr,ev=next(split.split(x,y)); xt,xe=x[tr],x[ev]; yt,ye=y[tr],y[ev]; d=x.shape[1]
        pop=self.rng.integers(0,2,size=(self.population,d),dtype=np.int8); empty=np.flatnonzero(pop.sum(1)==0)
        if empty.size: pop[empty,self.rng.integers(0,d,size=empty.size)]=1
        best=None; best_score=-1.0; history=[]
        def fitness(mask):
            sel=np.flatnonzero(mask); clf=SVC(kernel="rbf",gamma="scale",C=self.svm_c,cache_size=4096); clf.fit(xt[:,sel],yt)
            return float(accuracy_score(ye,clf.predict(xe[:,sel])))
        for generation in range(self.generations):
            scores=np.asarray([fitness(m) for m in pop]); order=np.argsort(-scores)
            if best is None or scores[order[0]]>best_score:
                best_score=float(scores[order[0]]); best=pop[order[0]].copy()
            history.append({"generation":generation+1,"best_inner_accuracy":float(scores[order[0]]),"global_best_inner_accuracy":best_score,"selected_features":int(best.sum())})
            parent_count=max(2,self.population//2); parents=pop[order[:parent_count]]
            offspring=np.empty((self.population-parent_count,d),dtype=np.int8); midpoint=max(1,d//2)
            for i in range(len(offspring)):
                p1=parents[i%parent_count]; p2=parents[(i+1)%parent_count]; offspring[i,:midpoint]=p1[:midpoint]; offspring[i,midpoint:]=p2[midpoint:]
                if self.mutation_count>0:
                    mutation_idx=self.rng.integers(0,d,size=self.mutation_count); offspring[i,mutation_idx]=1-offspring[i,mutation_idx]
                if offspring[i].sum()==0: offspring[i,self.rng.integers(0,d)]=1
            pop=np.vstack((parents,offspring))
        return np.flatnonzero(best).astype(np.int64),history
