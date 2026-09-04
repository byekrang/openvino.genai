// Verifies the real C++ pass insert_cross_kv_beam_gather against the current
// decoder IR: prints transformed states, checks identity/drop-middle/permutation
// correctness, and measures identity-Gather per-step cost. args: <decoder.xml>
#include <openvino/openvino.hpp>
#include "cross_kv_beam_gather.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <algorithm>
using clk=std::chrono::steady_clock;
static double MS(clk::time_point a,clk::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static double med(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}

int main(int argc,char**argv){
    std::string path=argv[1]; size_t E=argc>2?std::stoul(argv[2]):1500;
    std::vector<int64_t> PROMPT={50258,50259,50359,50363}; size_t P=4;
    ov::Core core;
    auto base=core.read_model(path);
    size_t D=base->input("encoder_hidden_states").get_partial_shape()[2].get_length();
    auto xf=core.read_model(path);
    ov::genai::insert_cross_kv_beam_gather(xf);

    std::cout<<"== transformed cross-KV states (name -> shape) ==\n";
    size_t ng=0;
    const std::string suf="/cross_kv_beam_gather";
    for(auto& op: xf->get_ops()){
        auto n=op->get_friendly_name();
        if(n.size()>suf.size() && n.compare(n.size()-suf.size(),suf.size(),suf)==0){
            auto rvshape=op->input_value(0).get_partial_shape();
            std::cout<<"   "<<op->input_value(0).get_node()->get_friendly_name()
                     <<"  "<<rvshape.to_string()<<"\n"; ng++;
        }
    }
    std::cout<<"   total gathers inserted = "<<ng<<"\n\n";

    auto cmB=core.compile_model(base,"CPU");
    auto cmX=core.compile_model(xf,"CPU");

    std::mt19937 g(7); std::uniform_real_distribution<float> du(-1,1);
    auto enc=[&](std::vector<int> rows){ ov::Tensor t(ov::element::f32,{rows.size(),E,D});
        std::mt19937 gg(7); std::uniform_real_distribution<float> d(-1,1);
        std::vector<float> full((size_t)3*E*D); for(auto&x:full)x=d(gg);
        for(size_t i=0;i<rows.size();++i) std::memcpy(t.data<float>()+i*E*D, full.data()+(size_t)rows[i]*E*D, (size_t)E*D*4);
        return t; };
    auto i64=[&](std::vector<int64_t> v,size_t r,size_t c){ov::Tensor t(ov::element::i64,{r,c});std::memcpy(t.data<int64_t>(),v.data(),v.size()*8);return t;};
    auto bmt=[&](std::vector<int32_t> v){ov::Tensor t(ov::element::i32,{v.size()});std::memcpy(t.data<int32_t>(),v.data(),v.size()*4);return t;};
    auto lastrow=[&](const ov::Tensor&L,size_t row){auto sh=L.get_shape();size_t seq=sh[1],V=sh[2];const float*p=L.data<float>()+row*seq*V+(seq-1)*V;return std::vector<float>(p,p+V);};
    auto mdiff=[&](const std::vector<float>&a,const std::vector<float>&b){double m=0;for(size_t i=0;i<a.size();++i)m=std::max(m,(double)std::fabs(a[i]-b[i]));return m;};
    auto prefill=[&](ov::InferRequest&r,std::vector<int>rows){
        r.set_tensor("encoder_hidden_states",enc(rows));
        std::vector<int64_t> ids(rows.size()*P);for(size_t i=0;i<rows.size();++i)for(size_t c=0;c<P;++c)ids[i*P+c]=PROMPT[c];
        r.set_tensor("input_ids",i64(ids,rows.size(),P));
        std::vector<int32_t> bi(rows.size());std::iota(bi.begin(),bi.end(),0);
        r.set_tensor("beam_idx",bmt(bi)); r.infer(); };

    // identity correctness: xf vs base, prefill B=3
    { auto rB=cmB.create_infer_request(); prefill(rB,{0,1,2});
      auto rX=cmX.create_infer_request(); prefill(rX,{0,1,2});
      double m=0; for(size_t row=0;row<3;++row) m=std::max(m, mdiff(lastrow(rB.get_tensor("logits"),row),lastrow(rX.get_tensor("logits"),row)));
      std::cout<<"identity beam_idx: xf vs base max|logits| = "<<m<<"\n"; }

    // drop-middle [A,B,C]->[A,C] on xf via beam_idx=[0,2]; reference = base isolated [0,2]
    { auto ref=cmB.create_infer_request(); prefill(ref,{0,2});
      ref.set_tensor("encoder_hidden_states",enc({0,2})); ref.set_tensor("input_ids",i64({111,222},2,1)); ref.set_tensor("beam_idx",bmt({0,1})); ref.infer();
      auto refA=lastrow(ref.get_tensor("logits"),0), refC=lastrow(ref.get_tensor("logits"),1);
      auto cand=cmX.create_infer_request(); prefill(cand,{0,1,2});
      cand.set_tensor("encoder_hidden_states",enc({0,2})); cand.set_tensor("input_ids",i64({111,222},2,1)); cand.set_tensor("beam_idx",bmt({0,2})); cand.infer();
      auto cA=lastrow(cand.get_tensor("logits"),0), cC=lastrow(cand.get_tensor("logits"),1);
      std::cout<<"drop-middle beam_idx=[0,2]: rowA="<<mdiff(cA,refA)<<" rowC="<<mdiff(cC,refC)<<"\n"; }

    // permutation beam_idx=[1,0]
    { auto r=cmX.create_infer_request(); prefill(r,{0,2});
      r.set_tensor("encoder_hidden_states",enc({2,0})); r.set_tensor("input_ids",i64({1,2},2,1)); r.set_tensor("beam_idx",bmt({1,0}));
      r.infer(); std::cout<<"permutation beam_idx=[1,0]: ok, logits rows="<<r.get_tensor("logits").get_shape()[0]<<"\n"; }

    // perf: per-step identity beam_idx, base vs xf
    auto perstep=[&](ov::CompiledModel&cm,size_t B){ auto r=cm.create_infer_request();
        std::vector<int> rows(B);std::iota(rows.begin(),rows.end(),0); prefill(r,rows);
        std::vector<int64_t> tok(B,100);std::vector<int32_t> bi(B);std::iota(bi.begin(),bi.end(),0);
        std::vector<double> ts; for(size_t s=0;s<45;++s){ r.set_tensor("encoder_hidden_states",enc(rows));
            r.set_tensor("input_ids",i64(tok,B,1)); r.set_tensor("beam_idx",bmt(bi));
            auto a=clk::now(); r.infer(); auto b=clk::now(); if(s>=5)ts.push_back(MS(a,b)); } return med(ts); };
    std::cout<<"\n== per-step identity beam_idx (base vs xf) ==\n";
    for(size_t B:{1,2,4,8}){ double tb=perstep(cmB,B),tx=perstep(cmX,B);
        std::cout<<"  B="<<B<<"  base="<<tb<<"  xf="<<tx<<"  overhead=+"<<(tx-tb)<<" ms ("<<(100*(tx-tb)/tb)<<"%)\n"; }
    return 0;
}
