import sys, time, json, numpy as np, librosa, openvino as ov
from transformers import WhisperFeatureExtractor

MODEL = sys.argv[1]
DEVICE = "CPU"
_gc = json.load(open(f"{MODEL}/generation_config.json"))
SOT = _gc["decoder_start_token_id"]
EOS = _gc["eos_token_id"]
NOTS = _gc.get("no_timestamps_token_id")
if _gc.get("is_multilingual", True):
    EN = _gc.get("lang_to_id",{}).get("<|en|>",50259)
    TRANSCRIBE = _gc.get("task_to_id",{}).get("transcribe",50359)
    PROMPT = [SOT,EN,TRANSCRIBE,NOTS]
else:
    PROMPT = [SOT,NOTS]
MAXNEW = 260

core = ov.Core()
enc_cm = core.compile_model(core.read_model(f"{MODEL}/openvino_encoder_model.xml"), DEVICE)
dec_cm = core.compile_model(core.read_model(f"{MODEL}/openvino_decoder_model.xml"), DEVICE)
feat = WhisperFeatureExtractor.from_pretrained(MODEL)
ENC_OUT = enc_cm.outputs[0].get_any_name()
ENC_IN  = enc_cm.inputs[0].get_any_name()

def encode(audio):
    f = feat(audio, sampling_rate=16000, return_tensors="np").input_features.astype(np.float32)
    r = enc_cm.create_infer_request()
    r.set_tensor(ENC_IN, ov.Tensor(f))
    r.infer()
    return r.get_tensor(ENC_OUT).data.copy()  # [1, E, D]

def scalar_decode(enc):
    r = dec_cm.create_infer_request()
    r.set_tensor("encoder_hidden_states", ov.Tensor(enc))
    r.set_tensor("input_ids", ov.Tensor(np.array([PROMPT],dtype=np.int64)))
    r.set_tensor("beam_idx", ov.Tensor(np.array([0],dtype=np.int32)))
    r.infer()
    tok = int(np.argmax(r.get_tensor("logits").data[0,-1,:]))
    out=[]
    for _ in range(MAXNEW):
        if tok==EOS: break
        out.append(tok)
        r.set_tensor("encoder_hidden_states", ov.Tensor(enc))
        r.set_tensor("input_ids", ov.Tensor(np.array([[tok]],dtype=np.int64)))
        r.set_tensor("beam_idx", ov.Tensor(np.array([0],dtype=np.int32)))
        r.infer()
        tok=int(np.argmax(r.get_tensor("logits").data[0,-1,:]))
    return out

def cross_state_mask(req, E):
    return [s.state.shape[2]==E for s in req.query_state()]

def batch_decode(encs, policy):
    """policy in {'filler','compact'}. Greedy. Returns dict with per-row tokens,
    finish steps, and timing (decoder infer wall, compaction wall, shrink count)."""
    B = len(encs)
    E = encs[0].shape[1]
    enc_full = np.concatenate(encs, axis=0)  # [B,E,D]
    req = dec_cm.create_infer_request()
    # prefill width B
    req.set_tensor("encoder_hidden_states", ov.Tensor(enc_full.copy()))
    req.set_tensor("input_ids", ov.Tensor(np.tile(PROMPT,(B,1)).astype(np.int64)))
    req.set_tensor("beam_idx", ov.Tensor(np.arange(B,dtype=np.int32)))
    req.infer()
    logits = req.get_tensor("logits").data
    tokens = [[] for _ in range(B)]
    finished = [False]*B
    finish_step = [None]*B
    # logical rows still active, in physical order
    active = list(range(B))                 # logical ids currently present physically
    enc_by_logical = {i: encs[i] for i in range(B)}
    # first tokens
    nxt = {i:int(np.argmax(logits[p,-1,:])) for p,i in enumerate(active)}
    t_infer=0.0; t_compact=0.0; shrinks=[]; step=0
    while True:
        # append/emit for active rows; detect finishers this step
        newly=[]
        for i in [j for j in range(B) if not finished[j] and j in nxt]:
            tk=nxt[i]
            if tk==EOS:
                finished[i]=True; finish_step[i]=step; newly.append(i)
            else:
                tokens[i].append(tk)
        if all(finished) or step>=MAXNEW:
            break
        if policy=='filler':
            # width stays B; finished rows carry filler token
            phys=list(range(B))
            ids=np.array([[nxt[i] if not finished[i] else SOT] for i in range(B)],dtype=np.int64)
            enc_in=enc_full
            beam=np.arange(B,dtype=np.int32)
            a=time.perf_counter()
            req.set_tensor("encoder_hidden_states", ov.Tensor(enc_in.copy()))
            req.set_tensor("input_ids", ov.Tensor(ids))
            req.set_tensor("beam_idx", ov.Tensor(beam)); req.infer()
            t_infer+=time.perf_counter()-a
            logits=req.get_tensor("logits").data
            nxt={i:int(np.argmax(logits[i,-1,:])) for i in range(B) if not finished[i]}
        else:  # compact
            survivors=[i for i in active if not finished[i]]
            if newly and survivors:
                # SHRINK: compact cross-KV to survivor physical positions
                surv_phys=[active.index(i) for i in survivors]
                c0=time.perf_counter()
                sts=req.query_state()
                pairs=[(s, np.array(s.state.data,copy=True)) for s in sts]  # single get per state
                for s,arr in pairs:
                    if arr.shape[2]==E:
                        s.state=ov.Tensor(np.ascontiguousarray(arr[surv_phys]))
                pairs=None; sts=None
                t_compact+=time.perf_counter()-c0
                shrinks.append(len(survivors))
                # next infer gathers self-attn by survivor phys, then reindex
                enc_full=np.concatenate([enc_by_logical[i] for i in survivors],axis=0)
                ids=np.array([[nxt[i]] for i in survivors],dtype=np.int64)
                beam=np.array(surv_phys,dtype=np.int32)
                active=survivors[:]
            else:
                ids=np.array([[nxt[i]] for i in survivors],dtype=np.int64)
                beam=np.arange(len(survivors),dtype=np.int32)
                active=survivors[:]
            if not survivors: break
            a=time.perf_counter()
            req.set_tensor("encoder_hidden_states", ov.Tensor(enc_full.copy()))
            req.set_tensor("input_ids", ov.Tensor(ids))
            req.set_tensor("beam_idx", ov.Tensor(beam)); req.infer()
            t_infer+=time.perf_counter()-a
            logits=req.get_tensor("logits").data
            nxt={i:int(np.argmax(logits[p,-1,:])) for p,i in enumerate(active)}
        step+=1
    round_len=step
    return dict(tokens=tokens, finish_step=finish_step, round_len=round_len,
                t_infer=t_infer, t_compact=t_compact, shrinks=shrinks, B=B)

def build_long_variants():
    import soundfile as sf
    y,_=librosa.load("/tmp/cand3_audio/x1.wav",sr=16000)
    for n in (6,8,12):
        sf.write(f"/tmp/cand3_audio/x{n}.wav", np.tile(y,n).astype(np.float32),16000)

if __name__=="__main__" and len(sys.argv)>2 and sys.argv[2]=="dump":
    # args: model dump <outdir> <m1,m2,...>
    import os, struct
    outdir=sys.argv[3]; members=sys.argv[4].split(",")
    os.makedirs(outdir,exist_ok=True)
    build_long_variants()
    for n in members:
        y,_=librosa.load(f"/tmp/cand3_audio/{n}.wav",sr=16000)
        e=encode(y)  # [1,E,D] float32
        _,E,D=e.shape
        with open(f"{outdir}/{n}.enc","wb") as f:
            f.write(struct.pack("<ii",E,D)); f.write(e.astype("<f4").tobytes())
        toks=scalar_decode(e)
        with open(f"{outdir}/{n}.tok","wb") as f:
            f.write(struct.pack("<i",len(toks)))
            f.write(np.array(toks,dtype="<i4").tobytes())
        print(f"dumped {n}: E={E} D={D} ntok={len(toks)}")
    with open(f"{outdir}/prompt.txt","w") as f:
        f.write(" ".join(map(str,PROMPT))+"\n"+str(EOS)+"\n")
    print("PROMPT",PROMPT,"EOS",EOS)

if __name__=="__main__" and len(sys.argv)>2 and sys.argv[2]=="validate":
    import openvino_genai as g
    pipe=g.WhisperPipeline(MODEL,"CPU")
    for name in ["x1","x2","tts1"]:
        y,_=librosa.load(f"/tmp/cand3_audio/{name}.wav",sr=16000)
        toks=scalar_decode(encode(y))
        gtxt=pipe.generate(y.tolist()).texts[0]
        print(f"{name}: harness_ntok={len(toks)} harness_toks={toks}")
        print(f"     genai={gtxt!r}")

if __name__=="__main__" and len(sys.argv)>2 and sys.argv[2]=="one":
    # args: model one <policy> <m1,m2,...>  -> JSON on stdout (isolated process)
    import json as _j
    policy=sys.argv[3]; members=sys.argv[4].split(",")
    encs=[]; scal=[]
    for n in members:
        y,_=librosa.load(f"/tmp/cand3_audio/{n}.wav",sr=16000)
        e=encode(y); encs.append(e); scal.append(scalar_decode(e))
    _=batch_decode(encs,'filler')  # warmup (safe, no set_state)
    if policy=='filler':
        IT=[]
        for _ in range(5):
            r=batch_decode(encs,'filler'); IT.append(r["t_infer"]*1000)
        IT.sort()
        r0=batch_decode(encs,'filler')
        corr=all(r0["tokens"][i]==scal[i] for i in range(len(members)))
        print("JSON"+_j.dumps(dict(policy=policy,members=members,finish_step=r0["finish_step"],
            round_len=r0["round_len"],t_infer=IT[len(IT)//2],shrinks=[],corr=corr)))
    else:  # compact: ONE timed round (plugin fragile across many set_state rounds)
        r0=batch_decode(encs,'compact')
        corr=all(r0["tokens"][i]==scal[i] for i in range(len(members)))
        print("JSON"+_j.dumps(dict(policy=policy,members=members,finish_step=r0["finish_step"],
            round_len=r0["round_len"],t_infer=r0["t_infer"]*1000,t_compact=r0["t_compact"]*1000,
            shrinks=r0["shrinks"],corr=corr)))
