// Candidate #3 cost microbench on the real whisper-tiny stateful decoder.
// Measures the one-time shrink-event cost (cross-KV batch compaction) broken
// into sub-ops, and the per-decode-step infer time at width 3 (filler) vs
// width 2 (compacted). No NumPy. Correctness is proven separately in Python.

#include <openvino/openvino.hpp>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <string>
#include <algorithm>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }
static double median(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }

static ov::Tensor rand_f32(const ov::Shape& sh,uint32_t seed){
    ov::Tensor t(ov::element::f32,sh); std::mt19937 g(seed); std::uniform_real_distribution<float> d(-1,1);
    float* p=t.data<float>(); for(size_t i=0;i<t.get_size();++i)p[i]=d(g); return t;
}
static ov::Tensor i64_t(const std::vector<int64_t>& v,size_t r,size_t c){
    ov::Tensor t(ov::element::i64,ov::Shape{r,c}); std::memcpy(t.data<int64_t>(),v.data(),v.size()*8); return t;
}
static ov::Tensor beam_t(const std::vector<int32_t>& v){
    ov::Tensor t(ov::element::i32,ov::Shape{v.size()}); std::memcpy(t.data<int32_t>(),v.data(),v.size()*4); return t;
}

int main(int argc,char** argv){
    // decoder IR: env WHISPER_DECODER, or argv[3], else a local default
    const char* env_mp = std::getenv("WHISPER_DECODER");
    std::string mp = argc>3 ? argv[3] : (env_mp ? env_mp : "openvino_decoder_model.xml");
    size_t E = argc>1?std::stoul(argv[1]):1500;
    size_t P = 4, D = 384, B = 3;
    size_t REP = argc>2?std::stoul(argv[2]):50;
    ov::Core core; auto compiled=core.compile_model(core.read_model(mp),"CPU");
    ov::Tensor enc=rand_f32(ov::Shape{B,E,D},123);

    auto enc_rows=[&](std::vector<int> rows){
        ov::Tensor t(ov::element::f32,ov::Shape{rows.size(),E,D});
        for(size_t i=0;i<rows.size();++i) std::memcpy(t.data<float>()+i*E*D, enc.data<float>()+(size_t)rows[i]*E*D, E*D*4);
        return t;
    };
    auto prefill=[&](ov::InferRequest& r,std::vector<int> rows){
        r.set_tensor("encoder_hidden_states",enc_rows(rows));
        std::vector<int64_t> ids(rows.size()*P); for(size_t i=0;i<rows.size();++i)for(size_t c=0;c<P;++c)ids[i*P+c]=1+c;
        r.set_tensor("input_ids",i64_t(ids,rows.size(),P));
        std::vector<int32_t> bi(rows.size()); std::iota(bi.begin(),bi.end(),0);
        r.set_tensor("beam_idx",beam_t(bi)); r.infer();
    };

    // ---- discover cross-attn states (seq-dim==E) once; report per-state bytes ----
    {
        ov::InferRequest r=compiled.create_infer_request(); prefill(r,{0,1,2});
        size_t ncross=0,cross_bytes_per_row=0; auto sts=r.query_state();
        for(auto& s:sts){ auto sh=s.get_state().get_shape(); if(sh[2]==E){ ncross++; cross_bytes_per_row=sh[1]*sh[2]*sh[3]*4; } }
        std::cout<<"E="<<E<<" cross_states="<<ncross<<" bytes/row/state="<<cross_bytes_per_row
                 <<" total_survivor_bytes(2 rows)="<<(double)ncross*cross_bytes_per_row*2/1e6<<" MB\n";
    }

    // ================= shrink-event sub-op microbench (drop-middle {0,2}) =========
    // Straightforward B: get_state(view) -> dense gather survivors -> set_state.
    std::vector<double> vget,vgather,vset,vtot;
    for(size_t rep=0;rep<REP;++rep){
        ov::InferRequest r=compiled.create_infer_request(); prefill(r,{0,1,2});
        auto sts=r.query_state();
        std::vector<bool> cross; std::vector<ov::Shape> shp;
        for(auto& s:sts){ auto sh=s.get_state().get_shape(); cross.push_back(sh[2]==E); shp.push_back(sh);}
        double tget=0,tgat=0,tset=0; auto tot0=clk::now();
        for(size_t i=0;i<sts.size();++i){
            if(!cross[i]) continue;
            auto g0=clk::now(); ov::Tensor old=sts[i].get_state();
            // copy out immediately (required: view aliases live plugin memory)
            size_t re=shp[i][1]*shp[i][2]*shp[i][3];
            ov::Tensor compact(ov::element::f32,ov::Shape{2,shp[i][1],shp[i][2],shp[i][3]});
            auto g1=clk::now(); tget+=ms(g0,g1);
            auto a0=clk::now();
            const float* src=old.data<float>(); float* dst=compact.data<float>();
            std::memcpy(dst, src+0*re, re*4);        // row 0
            std::memcpy(dst+re, src+2*re, re*4);     // row 2
            auto a1=clk::now(); tgat+=ms(a0,a1);
            auto s0=clk::now(); sts[i].set_state(compact); auto s1=clk::now(); tset+=ms(s0,s1);
        }
        auto tot1=clk::now();
        vget.push_back(tget); vgather.push_back(tgat); vset.push_back(tset); vtot.push_back(ms(tot0,tot1));
    }
    std::cout<<"\n== SHRINK EVENT (C++ straightforward B, drop-middle, medians over "<<REP<<" reps) ==\n";
    std::cout<<"  get_state (8)     : "<<median(vget)   <<" ms\n";
    std::cout<<"  gather/alloc (8)  : "<<median(vgather)<<" ms\n";
    std::cout<<"  set_state (8)     : "<<median(vset)   <<" ms\n";
    std::cout<<"  SHRINK TOTAL      : "<<median(vtot)   <<" ms\n";

    // ================= per-decode-step infer cost: width 3 vs width 2 ============
    auto steady=[&](std::vector<int> rows,std::vector<int32_t> beam){
        ov::InferRequest r=compiled.create_infer_request(); prefill(r,rows);
        std::vector<int64_t> tok(rows.size(),100);
        std::vector<double> t;
        for(size_t s=0;s<REP+5;++s){
            r.set_tensor("encoder_hidden_states",enc_rows(rows));
            r.set_tensor("input_ids",i64_t(tok,rows.size(),1));
            r.set_tensor("beam_idx",beam_t(beam));
            auto a=clk::now(); r.infer(); auto b=clk::now();
            if(s>=5) t.push_back(ms(a,b));
        }
        return median(t);
    };
    double w3=steady({0,1,2},{0,1,2});
    double w2=steady({0,2},{0,1});
    std::cout<<"\n== PER-STEP INFER (median over "<<REP<<") ==\n";
    std::cout<<"  width-3 (filler)  : "<<w3<<" ms/step\n";
    std::cout<<"  width-2 (compacted): "<<w2<<" ms/step\n";
    double save=w3-w2;
    std::cout<<"  per-step saving   : "<<save<<" ms/step\n";
    if(save>0) std::cout<<"  BREAK-EVEN remaining steps = shrink_total/save = "
                        <<median(vtot)/save<<" steps\n";
    else std::cout<<"  per-step saving <= 0 : compaction does not pay off at this size\n";
    return 0;
}
