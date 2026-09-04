// Integrated Candidate #3 experiment: full decoder round in C++ (prefill ->
// autoregressive greedy -> detect finished rows -> compact cross-KV via the
// optimized C++ VariableState copy-out path -> compacting beam_idx for self-KV
// -> continue -> multiple shrink events). Directly measures whole-round wall
// time for filler vs compact. No NumPy/Python in the state path.
//
// args: <decoder.xml> <enc_dir> <m1,m2,...> <soak_iters>

#include <openvino/openvino.hpp>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

using clk = std::chrono::steady_clock;
static double MS(clk::time_point a, clk::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }
static double median(std::vector<double> v){ std::sort(v.begin(),v.end()); return v.empty()?0:v[v.size()/2]; }

struct Enc { int E,D; std::vector<float> data; }; // [1,E,D]

static Enc load_enc(const std::string& path){
    std::ifstream f(path, std::ios::binary); int E,D;
    f.read((char*)&E,4); f.read((char*)&D,4);
    Enc e; e.E=E; e.D=D; e.data.resize((size_t)E*D);
    f.read((char*)e.data.data(), (size_t)E*D*4); return e;
}
static std::vector<int> load_tok(const std::string& path){
    std::ifstream f(path, std::ios::binary); int n; f.read((char*)&n,4);
    std::vector<int32_t> t(n); f.read((char*)t.data(), n*4);
    return std::vector<int>(t.begin(), t.end());
}
static std::vector<std::string> split(const std::string& s,char d){
    std::vector<std::string> o; std::stringstream ss(s); std::string x;
    while(std::getline(ss,x,d)) o.push_back(x); return o;
}

int main(int argc,char** argv){
    std::string dec=argv[1], encdir=argv[2];
    auto members=split(argv[3],',');
    int SOAK = argc>4?std::stoi(argv[4]):1;
    const size_t MAXST=320;

    // prompt + eos
    std::vector<int64_t> PROMPT; int64_t EOS;
    { std::ifstream f(encdir+"/prompt.txt"); std::string line; std::getline(f,line);
      for(auto&t:split(line,' ')) PROMPT.push_back(std::stoll(t));
      std::string es; std::getline(f,es); EOS=std::stoll(es); }
    size_t P=PROMPT.size();

    ov::Core core; auto compiled=core.compile_model(core.read_model(dec),"CPU");

    // load encoder states + scalar refs
    std::vector<Enc> enc; std::vector<std::vector<int>> ref;
    for(auto&m:members){ enc.push_back(load_enc(encdir+"/"+m+".enc")); ref.push_back(load_tok(encdir+"/"+m+".tok")); }
    size_t B=members.size(), E=enc[0].E, D=enc[0].D;

    auto beam_t=[&](const std::vector<int32_t>& v){ ov::Tensor t(ov::element::i32,ov::Shape{v.size()}); std::memcpy(t.data<int32_t>(),v.data(),v.size()*4); return t; };
    auto enc_of=[&](const std::vector<int>& ids){ // logical ids -> [W,E,D]
        ov::Tensor t(ov::element::f32,ov::Shape{ids.size(),E,D});
        for(size_t i=0;i<ids.size();++i) std::memcpy(t.data<float>()+i*E*D, enc[ids[i]].data.data(), (size_t)E*D*4);
        return t; };
    auto argmax=[&](const ov::Tensor& L,size_t row){ auto sh=L.get_shape(); size_t seq=sh[1],V=sh[2];
        const float* p=L.data<float>()+row*seq*V+(seq-1)*V; return (int64_t)std::distance(p,std::max_element(p,p+V)); };

    // returns tokens per logical row; fills timings + traces
    auto run=[&](bool compact, double& t_infer,double& t_compact,
                 std::vector<int>& shrink_w, std::vector<int>& width_trace, double& wall)->std::vector<std::vector<int>>{
        t_infer=0;t_compact=0; shrink_w.clear(); width_trace.clear();
        auto W0=clk::now();
        ov::InferRequest req=compiled.create_infer_request();
        std::vector<int> all(B); std::iota(all.begin(),all.end(),0);
        // prefill
        { std::vector<int64_t> ids(B*P); for(size_t i=0;i<B;++i) for(size_t c=0;c<P;++c) ids[i*P+c]=PROMPT[c];
          ov::Tensor t(ov::element::i64,ov::Shape{B,P}); std::memcpy(t.data<int64_t>(),ids.data(),ids.size()*8);
          std::vector<int32_t> bi(B); std::iota(bi.begin(),bi.end(),0);
          req.set_tensor("encoder_hidden_states",enc_of(all)); req.set_tensor("input_ids",t); req.set_tensor("beam_idx",beam_t(bi));
          auto a=clk::now(); req.infer(); t_infer+=MS(a,clk::now()); }
        auto L=req.get_tensor("logits");
        std::vector<std::vector<int>> tok(B); std::vector<char> fin(B,0); std::vector<int> fstep(B,-1);
        std::vector<int> active(all);
        std::map<int,int64_t> nxt; for(size_t p=0;p<active.size();++p) nxt[active[p]]=argmax(L,p);
        size_t step=0;
        while(true){
            std::vector<int> newly;
            for(int i=0;i<(int)B;++i){ if(fin[i]||!nxt.count(i))continue; int64_t tk=nxt[i];
                if(tk==EOS){ fin[i]=1; fstep[i]=step; newly.push_back(i);} else tok[i].push_back((int)tk); }
            bool allfin=std::all_of(fin.begin(),fin.end(),[](char c){return c;});
            if(allfin||step>=MAXST) break;
            if(!compact){
                width_trace.push_back((int)B);
                std::vector<int64_t> ids(B); for(size_t i=0;i<B;++i) ids[i]= fin[i]?PROMPT[0]:nxt[i];
                ov::Tensor t(ov::element::i64,ov::Shape{B,1}); std::memcpy(t.data<int64_t>(),ids.data(),B*8);
                std::vector<int32_t> bi(B); std::iota(bi.begin(),bi.end(),0);
                req.set_tensor("encoder_hidden_states",enc_of(all)); req.set_tensor("input_ids",t); req.set_tensor("beam_idx",beam_t(bi));
                auto a=clk::now(); req.infer(); t_infer+=MS(a,clk::now());
                L=req.get_tensor("logits"); nxt.clear();
                for(int i=0;i<(int)B;++i) if(!fin[i]) nxt[i]=argmax(L,i);
            } else {
                std::vector<int> surv; for(int i:active) if(!fin[i]) surv.push_back(i);
                if(surv.empty()) break;
                std::vector<int32_t> beam;
                if(!newly.empty()){
                    std::vector<int32_t> surv_phys;
                    for(int s:surv){ surv_phys.push_back((int32_t)(std::find(active.begin(),active.end(),s)-active.begin())); }
                    // ---- optimized C++ cross-KV compaction (copy-out, no retained view) ----
                    auto c0=clk::now();
                    auto states=req.query_state();
                    for(auto& st:states){
                        ov::Tensor cur=st.get_state(); auto sh=cur.get_shape();
                        if(sh[2]!=E) continue;               // cross-attn only
                        size_t re=sh[1]*sh[2]*sh[3];
                        ov::Tensor comp(ov::element::f32,ov::Shape{surv_phys.size(),sh[1],sh[2],sh[3]});
                        const float* src=cur.data<float>(); float* dst=comp.data<float>();
                        for(size_t r=0;r<surv_phys.size();++r) std::memcpy(dst+r*re, src+(size_t)surv_phys[r]*re, re*4);
                        st.set_state(comp);                  // cur (view) not used again
                    }
                    states.clear();
                    t_compact+=MS(c0,clk::now());
                    shrink_w.push_back((int)surv.size());
                    beam=surv_phys; active=surv;
                } else { beam.resize(surv.size()); std::iota(beam.begin(),beam.end(),0); active=surv; }
                width_trace.push_back((int)active.size());
                std::vector<int64_t> ids(active.size()); for(size_t p=0;p<active.size();++p) ids[p]=nxt[active[p]];
                ov::Tensor t(ov::element::i64,ov::Shape{active.size(),1}); std::memcpy(t.data<int64_t>(),ids.data(),ids.size()*8);
                req.set_tensor("encoder_hidden_states",enc_of(active)); req.set_tensor("input_ids",t); req.set_tensor("beam_idx",beam_t(beam));
                auto a=clk::now(); req.infer(); t_infer+=MS(a,clk::now());
                L=req.get_tensor("logits"); nxt.clear();
                for(size_t p=0;p<active.size();++p) nxt[active[p]]=argmax(L,p);
            }
            step++;
        }
        wall=MS(W0,clk::now());
        return tok;
    };

    auto check=[&](const std::vector<std::vector<int>>& tk){ for(size_t i=0;i<B;++i) if(tk[i]!=ref[i]) return false; return true; };

    // warmup
    { double a,b,w; std::vector<int> s,wt; run(false,a,b,s,wt,w); }

    // timed medians over reps for each policy (fresh request each round)
    std::vector<double> fillW, compW, compI, compC;
    std::vector<int> shrinks, wtrace; double ti,tc,wall;
    std::vector<std::vector<int>> tkF, tkC;
    int REPS  = argc>5?std::stoi(argv[5]):7;   // filler reps
    int CREPS = argc>6?std::stoi(argv[6]):REPS; // compact reps
    for(int r=0;r<REPS;++r){ auto t=run(false,ti,tc,shrinks,wtrace,wall); fillW.push_back(wall); if(r==0)tkF=t; }
    for(int r=0;r<CREPS;++r){ auto t=run(true, ti,tc,shrinks,wtrace,wall); compW.push_back(wall); compI.push_back(ti); compC.push_back(tc); if(r==0)tkC=t; }

    bool okF=check(tkF), okC=check(tkC);
    bool AB=true; for(size_t i=0;i<B;++i) if(tkF[i]!=tkC[i]) AB=false;

    std::cout<<"BATCH "<<argv[3]<<"  B="<<B<<" E="<<E<<" D="<<D<<"\n";
    std::cout<<"  shrinks(widths)="; for(int x:shrinks)std::cout<<x<<" "; std::cout<<"\n";
    std::cout<<"  FILLER  round_wall(med)="<<median(fillW)<<" ms\n";
    std::cout<<"  COMPACT round_wall(med)="<<median(compW)<<" ms  [infer(med)="<<median(compI)
             <<" compaction(med)="<<median(compC)<<"]\n";
    std::cout<<"  net (COMPACT-FILLER) = "<<(median(compW)-median(fillW))<<" ms  ["
             <<(median(compW)<median(fillW)?"B wins":"A wins")<<"]\n";
    std::cout<<"  correctness: fillerVSscalar="<<(okF?"OK":"FAIL")
             <<" compactVSscalar="<<(okC?"OK":"FAIL")<<" filler==compact="<<(AB?"OK":"FAIL")<<"\n";

    // ---- soak: repeated compact rounds in ONE process ----
    if(SOAK>1){
        bool stable=true; std::vector<std::vector<int>> first;
        for(int it=0;it<SOAK;++it){ double a,b,w; std::vector<int> s,wt;
            auto t=run(true,a,b,s,wt,w);
            if(it==0) first=t; else { for(size_t i=0;i<B;++i) if(t[i]!=first[i]) stable=false; }
            if(!check(t)) stable=false;
        }
        std::cout<<"  SOAK("<<SOAK<<" compact rounds, one process): "
                 <<(stable?"STABLE (no crash, outputs identical & correct)":"UNSTABLE")<<"\n";
    }
    std::cout<<"\n";
    return 0;
}
