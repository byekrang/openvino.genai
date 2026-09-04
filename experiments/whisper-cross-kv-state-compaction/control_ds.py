import numpy as np, openvino as ov, sys
core=ov.Core(); cm=core.compile_model(core.read_model(sys.argv[1]),"CPU")
E,D=1500,768; PROMPT=[50257,50362]  # distil-small.en prompt
def enc(B): return ov.Tensor((0.1*np.ones((B,E,D))).astype(np.float32))
def rnd():
    r=cm.create_infer_request()
    r.set_tensor("encoder_hidden_states",enc(4)); r.set_tensor("input_ids",ov.Tensor(np.tile(PROMPT,(4,1)).astype(np.int64)))
    r.set_tensor("beam_idx",ov.Tensor(np.arange(4,dtype=np.int32))); r.infer()
    sts=r.query_state(); pairs=[(s,np.array(s.state.data,copy=True)) for s in sts]
    for s,a in pairs:
        if a.shape[2]==E: s.state=ov.Tensor(np.ascontiguousarray(a[[0,1,2]]))
    pairs=None; sts=None
    r.set_tensor("encoder_hidden_states",enc(3)); r.set_tensor("input_ids",ov.Tensor(np.full((3,1),100,dtype=np.int64)))
    r.set_tensor("beam_idx",ov.Tensor(np.array([0,1,2],dtype=np.int32))); r.infer(); del r
for i in range(3): rnd(); print(f"  round {i+1} ok",flush=True)
print("distil COMPLETED",flush=True)
