#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np, torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torchvision.models import ResNet50_Weights,VGG16_Weights,resnet50,vgg16
from experiments.sipakmed.paper_baselines_v1.train_mtfm_sipakmed import image_tensor

class ResNet50VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        r=resnet50(weights=ResNet50_Weights.DEFAULT); v=vgg16(weights=VGG16_Weights.DEFAULT)
        self.r=nn.Sequential(r.conv1,r.bn1,r.relu,r.maxpool,r.layer1,r.layer2,r.layer3,r.layer4)
        self.v=v.features
        for p in self.parameters(): p.requires_grad_(False)
    @torch.inference_mode()
    def forward(self,x):
        return torch.cat((self.r(x).flatten(1),self.v(x).flatten(1)),1)

class FeatureImageDataset(torch.utils.data.Dataset):
    def __init__(self,paths): self.paths=paths
    def __len__(self): return len(self.paths)
    def __getitem__(self,i): return image_tensor(self.paths[i],"identity"),i

def _write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); tmp.replace(path)

def _signature(paths):
    h=hashlib.sha256()
    for p in paths: h.update(p.encode()); h.update(b"\n")
    return h.hexdigest()

def extract_or_load_features(paths,cache_root,device,batch_size=8,workers=0):
    cache_root.mkdir(parents=True,exist_ok=True); fp=cache_root/"resnet50_vgg16_features.npy"; mp=cache_root/"metadata.json"; sig=_signature(paths)
    if fp.is_file() and mp.is_file():
        m=json.loads(mp.read_text(encoding="utf-8"))
        if m.get("path_signature")==sig and m.get("count")==len(paths): return np.load(fp,mmap_mode="r")
    loader=torch.utils.data.DataLoader(FeatureImageDataset(paths),batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=device.type=="cuda",persistent_workers=workers>0)
    model=ResNet50VGG16FeatureExtractor().to(device).eval(); out=None; tmp=cache_root/"resnet50_vgg16_features.tmp.npy"
    if tmp.exists(): tmp.unlink()
    for images,indices in loader:
        f=model(images.to(device,non_blocking=True)).float().cpu().numpy()
        if out is None: out=np.lib.format.open_memmap(tmp,mode="w+",dtype=np.float32,shape=(len(paths),int(f.shape[1])))
        out[indices.numpy()]=f
    if out is None: raise RuntimeError("no images")
    out.flush(); dim=int(f.shape[1]); del out,model
    if device.type=="cuda": torch.cuda.empty_cache()
    tmp.replace(fp); _write_json(mp,{"method":"PCA-GWO-ResNet50-VGG16-RBF-SVM","feature_extractors":["ResNet-50","VGG-16"],"feature_representation":"flattened penultimate convolutional feature maps","count":len(paths),"dimension":dim,"path_signature":sig,"weights":"ImageNet pretrained torchvision weights"})
    return np.load(fp,mmap_mode="r")

def fit_pca_99(x,variance_target=0.99,max_components=1024,seed=42):
    n=min(int(max_components),x.shape[0]-1,x.shape[1])
    if n<2: raise ValueError("PCA needs at least two components")
    p=PCA(n_components=n,svd_solver="randomized",random_state=seed); p.fit(x); cum=np.cumsum(p.explained_variance_ratio_)
    hit=np.flatnonzero(cum>=variance_target); k=int(hit[0]+1) if len(hit) else n
    return p,k,float(cum[k-1])

def transform_pca(p,k,x): return np.asarray(p.transform(x)[:,:k],dtype=np.float32)

class BinaryGWO:
    def __init__(self,population,iterations,seed):
        if population<4 or iterations<1: raise ValueError("population>=4 and iterations>=1 required")
        self.population=int(population); self.iterations=int(iterations); self.rng=np.random.default_rng(seed)
    def optimize(self,x,y,inner_fraction=0.25):
        sr=np.random.default_rng(int(self.rng.integers(0,2**31-1))); tr=[]; ev=[]
        for label in sorted(np.unique(y).tolist()):
            ix=np.flatnonzero(y==label); sr.shuffle(ix); cut=max(1,int(round(len(ix)*(1-inner_fraction)))); tr+=ix[:cut].tolist(); ev+=ix[cut:].tolist()
        tr=np.asarray(tr); ev=np.asarray(ev); xt,yt=x[tr],y[tr]; xe,ye=x[ev],y[ev]; d=x.shape[1]
        pos=self.rng.integers(0,2,size=(self.population,d),dtype=np.int8); empty=np.flatnonzero(pos.sum(1)==0)
        if empty.size: pos[empty,self.rng.integers(0,d,size=empty.size)]=1
        alpha=beta=delta=None; history=[]
        def score(mask):
            sel=np.flatnonzero(mask); knn=KNeighborsClassifier(n_neighbors=5,n_jobs=-1); knn.fit(xt[:,sel],yt)
            acc=float(accuracy_score(ye,knn.predict(xe[:,sel]))); return .99*(1-acc)+.01*len(sel)/d
        for it in range(self.iterations):
            scores=np.asarray([score(m) for m in pos]); order=np.argsort(scores); alpha,beta,delta=(pos[i].copy() for i in order[:3])
            history.append({"iteration":it+1,"alpha_fitness":float(scores[order[0]]),"selected_features":int(alpha.sum())})
            a=2-2*it/max(1,self.iterations-1); nxt=np.zeros_like(pos)
            for w in range(self.population):
                votes=np.zeros(d,dtype=np.float32)
                for leader in (alpha,beta,delta):
                    r1=self.rng.random(d); r2=self.rng.random(d); A=2*a*r1-a; C=2*r2; D=np.abs(C*leader.astype(np.float32)-pos[w]); z=leader.astype(np.float32)-A*D
                    votes+=1/(1+np.exp(-np.clip(z,-30,30)))
                nxt[w]=(self.rng.random(d)<votes/3).astype(np.int8)
            empty=np.flatnonzero(nxt.sum(1)==0)
            if empty.size: nxt[empty,self.rng.integers(0,d,size=empty.size)]=1
            pos=nxt
        if alpha is None: raise RuntimeError("GWO did not evaluate")
        return np.flatnonzero(alpha).astype(np.int64),history
