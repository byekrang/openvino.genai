import numpy as np, openvino as ov, sys
mode=sys.argv[2]; N=int(sys.argv[3])
core=ov.Core(); cm=core.compile_model(core.read_model(sys.argv[1]),"CPU")
E,D,P=1500,384,4; PROMPT=[50258,50259,50359,50363]
def enc(B): return ov.Tensor((0.1*np.ones((B,E,D))).astype(np.float32))
def rnd():
    r=cm.create_infer_request()
    r.set_tensor("encoder_hidden_states",enc(4)); r.set_tensor("input_ids",ov.Tensor(np.tile(PROMPT,(4,1)).astype(np.int64)))
    r.set_tensor("beam_idx",ov.Tensor(np.arange(4,dtype=np.int32))); r.infer()
    if mode!="none":
        sts=r.query_state(); pairs=[(s,np.array(s.state.data,copy=True)) for s in sts]
        for s,a in pairs:
            if a.shape[2]==E:
                sel=[0,1,2] if mode=="shrink" else [0,1,2,3]   # shrink vs same-shape set_state
                s.state=ov.Tensor(np.ascontiguousarray(a[sel]))
        pairs=None; sts=None
    W=3 if mode=="shrink" else 4
    r.set_tensor("encoder_hidden_states",enc(W)); r.set_tensor("input_ids",ov.Tensor(np.full((W,1),100,dtype=np.int64)))
    r.set_tensor("beam_idx",ov.Tensor(np.arange(W,dtype=np.int32))); r.infer()
    del r
for i in range(N): rnd(); print(f"  round {i+1} ok",flush=True)
print(f"MODE={mode} N={N} COMPLETED",flush=True)
