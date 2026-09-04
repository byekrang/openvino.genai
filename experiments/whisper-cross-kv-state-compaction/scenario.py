import numpy as np, openvino as ov, sys, os
scenario = sys.argv[1]
core = ov.Core()
# decoder IR: env WHISPER_DECODER, or argv[2], else a local default
model_path = os.environ.get("WHISPER_DECODER") or (sys.argv[2] if len(sys.argv) > 2 else "openvino_decoder_model.xml")
m = core.read_model(model_path)
cm = core.compile_model(m, "CPU")
E,D,B,P,STEPS = 50,384,3,4,8
enc = np.random.RandomState(1).randn(B,E,D).astype(np.float32)
sA=[10,20,30,40,50,60,70,80]; sC=[15,25,35,45,55,65,75,85]; FILLER=50258

def prefill(req, rows):
    req.set_tensor("encoder_hidden_states", ov.Tensor(enc[rows].copy()))
    req.set_tensor("input_ids", ov.Tensor(np.tile(np.arange(1,P+1),(len(rows),1)).astype(np.int64)))
    req.set_tensor("beam_idx", ov.Tensor(np.arange(len(rows),dtype=np.int32))); req.infer()
def lr(req,row): return req.get_tensor("logits").data[row,-1,:].copy()

def compact_cross(req, survivors):
    # single pass: for each state get -> materialize copy -> set; hold no view
    states = req.query_state()
    is_cross = []
    fulls = []
    for s in states:
        arr = np.array(s.state.data, copy=True)   # full materialized copy
        is_cross.append(arr.shape[2] == E)
        fulls.append(arr)
    for s, cross, arr in zip(states, is_cross, fulls):
        if cross:
            s.state = ov.Tensor(np.ascontiguousarray(arr[survivors]))
    states = None

if scenario == "ref":
    req=cm.create_infer_request(); prefill(req,[0,2]); A=[];C=[]
    for s in range(STEPS):
        req.set_tensor("encoder_hidden_states", ov.Tensor(enc[[0,2]].copy()))
        req.set_tensor("input_ids", ov.Tensor(np.array([[sA[s]],[sC[s]]],dtype=np.int64)))
        req.set_tensor("beam_idx", ov.Tensor(np.array([0,1],dtype=np.int32))); req.infer()
        A.append(lr(req,0)); C.append(lr(req,1))
elif scenario == "filler":
    req=cm.create_infer_request(); prefill(req,[0,1,2]); A=[];C=[]
    for s in range(STEPS):
        req.set_tensor("encoder_hidden_states", ov.Tensor(enc.copy()))
        req.set_tensor("input_ids", ov.Tensor(np.array([[sA[s]],[FILLER],[sC[s]]],dtype=np.int64)))
        req.set_tensor("beam_idx", ov.Tensor(np.array([0,1,2],dtype=np.int32))); req.infer()
        A.append(lr(req,0)); C.append(lr(req,2))
elif scenario in ("c3correct","c3neg"):
    survivors = [0,2] if scenario=="c3correct" else [0,1]
    first_beam = survivors
    req=cm.create_infer_request(); prefill(req,[0,1,2])
    compact_cross(req, survivors)
    A=[];C=[]
    for st in range(STEPS):
        toks=[[sC[st]] if sv==2 else [sA[st]] for sv in survivors]
        beam = first_beam if st==0 else list(range(len(survivors)))
        req.set_tensor("encoder_hidden_states", ov.Tensor(enc[survivors].copy()))
        req.set_tensor("input_ids", ov.Tensor(np.array(toks,dtype=np.int64)))
        req.set_tensor("beam_idx", ov.Tensor(np.array(beam,dtype=np.int32))); req.infer()
        A.append(lr(req,0)); C.append(lr(req,1))
else:
    raise SystemExit("unknown scenario")

np.save(f"/tmp/cand3_exp/{scenario}_A.npy", np.array(A))
np.save(f"/tmp/cand3_exp/{scenario}_C.npy", np.array(C))
print(f"{scenario}: saved logits A{np.array(A).shape} C{np.array(C).shape}")
