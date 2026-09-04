// Candidate #3 model-size scaling microbench. Production-style C++ (no NumPy).
// argv: <model.xml> <E> <B> <REP>. Drops the middle-ish finished row (index 1),
// survivors = all rows except 1 (non-contiguous for B>=3). Derives D from model.

#include <openvino/openvino.hpp>
#include <chrono>
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
    float* p=t.data<float>(); for(size_t i=0;i<t.get_size();++i)p[i]=d(g); return t; }
static ov::Tensor i64_t(const std::vector<int64_t>& v,size_t r,size_t c){
    ov::Tensor t(ov::element::i64,ov::Shape{r,c}); std::memcpy(t.data<int64_t>(),v.data(),v.size()*8); return t; }
static ov::Tensor beam_t(const std::vector<int32_t>& v){
    ov::Tensor t(ov::element::i32,ov::Shape{v.size()}); std::memcpy(t.data<int32_t>(),v.data(),v.size()*4); return t; }

int main(int argc,char** argv){
    std::string mp = argv[1];
    size_t E = argc>2?std::stoul(argv[2]):1500;
    size_t B = argc>3?std::stoul(argv[3]):3;
    size_t REP = argc>4?std::stoul(argv[4]):50;
    size_t P = 4;
    ov::Core core; auto model=core.read_model(mp); auto compiled=core.compile_model(model,"CPU");
    size_t D = compiled.input("encoder_hidden_states").get_partial_shape()[2].get_length();

    // survivors = all rows except index 1
    std::vector<int> surv; for(size_t i=0;i<B;++i) if(i!=1) surv.push_back((int)i);
    ov::Tensor enc=rand_f32(ov::Shape{B,E,D},123);
    auto enc_rows=[&](const std::vector<int>& rows){
        ov::Tensor t(ov::element::f32,ov::Shape{rows.size(),E,D});
        for(size_t i=0;i<rows.size();++i) std::memcpy(t.data<float>()+i*E*D, enc.data<float>()+(size_t)rows[i]*E*D, E*D*4);
        return t; };
    auto prefill=[&](ov::InferRequest& r,const std::vector<int>& rows){
        r.set_tensor("encoder_hidden_states",enc_rows(rows));
        std::vector<int64_t> ids(rows.size()*P); for(size_t i=0;i<rows.size();++i)for(size_t c=0;c<P;++c)ids[i*P+c]=1+c;
        r.set_tensor("input_ids",i64_t(ids,rows.size(),P));
        std::vector<int32_t> bi(rows.size()); std::iota(bi.begin(),bi.end(),0);
        r.set_tensor("beam_idx",beam_t(bi)); r.infer(); };

    std::vector<int> allrows(B); std::iota(allrows.begin(),allrows.end(),0);

    // report cross-state geometry
    size_t ncross=0,bpr=0;
    { ov::InferRequest r=compiled.create_infer_request(); prefill(r,allrows);
      for(auto& s:r.query_state()){ auto sh=s.get_state().get_shape(); if(sh[2]==E){ncross++;bpr=sh[1]*sh[2]*sh[3]*4;} } }
    double surv_MB=(double)ncross*bpr*surv.size()/1e6;
    std::cout<<mp<<"\n  D="<<D<<" E="<<E<<" B="<<B<<" cross_states="<<ncross
             <<" bytes/row/state="<<bpr<<" survivor_bytes="<<surv_MB<<" MB\n";

    // shrink event: compact cross states to survivors (drop row 1)
    std::vector<double> vget,vgat,vset,vtot;
    for(size_t rep=0;rep<REP;++rep){
        ov::InferRequest r=compiled.create_infer_request(); prefill(r,allrows);
        auto sts=r.query_state(); std::vector<ov::Shape> shp(sts.size()); std::vector<char> cross(sts.size());
        for(size_t i=0;i<sts.size();++i){ shp[i]=sts[i].get_state().get_shape(); cross[i]=(shp[i][2]==E); }
        double tg=0,ta=0,tsv=0; auto t0=clk::now();
        for(size_t i=0;i<sts.size();++i){ if(!cross[i])continue;
            auto g0=clk::now(); ov::Tensor old=sts[i].get_state();
            size_t re=shp[i][1]*shp[i][2]*shp[i][3];
            ov::Tensor comp(ov::element::f32,ov::Shape{surv.size(),shp[i][1],shp[i][2],shp[i][3]});
            auto g1=clk::now(); tg+=ms(g0,g1);
            auto a0=clk::now(); const float* src=old.data<float>(); float* dst=comp.data<float>();
            for(size_t r2=0;r2<surv.size();++r2) std::memcpy(dst+r2*re, src+(size_t)surv[r2]*re, re*4);
            auto a1=clk::now(); ta+=ms(a0,a1);
            auto s0=clk::now(); sts[i].set_state(comp); auto s1=clk::now(); tsv+=ms(s0,s1);
        }
        auto t1=clk::now(); vget.push_back(tg);vgat.push_back(ta);vset.push_back(tsv);vtot.push_back(ms(t0,t1));
    }

    // per-step infer: width-B (filler) vs width-(B-1) (survivors)
    auto steady=[&](const std::vector<int>& rows){
        ov::InferRequest r=compiled.create_infer_request(); prefill(r,rows);
        std::vector<int64_t> tok(rows.size(),100); std::vector<int32_t> bi(rows.size()); std::iota(bi.begin(),bi.end(),0);
        std::vector<double> t;
        for(size_t s=0;s<REP+5;++s){ r.set_tensor("encoder_hidden_states",enc_rows(rows));
            r.set_tensor("input_ids",i64_t(tok,rows.size(),1)); r.set_tensor("beam_idx",beam_t(bi));
            auto a=clk::now(); r.infer(); auto b=clk::now(); if(s>=5)t.push_back(ms(a,b)); }
        return median(t); };
    double wB=steady(allrows), wS=steady(surv), save=wB-wS;
    std::cout<<"  shrink: get="<<median(vget)<<" gather="<<median(vgat)<<" set="<<median(vset)
             <<" TOTAL="<<median(vtot)<<" ms\n";
    std::cout<<"  per-step: width"<<B<<"(filler)="<<wB<<" width"<<surv.size()<<"(compact)="<<wS
             <<" saving="<<save<<" ms/step\n";
    std::cout<<"  BREAK-EVEN = "<<(save>0?std::to_string(median(vtot)/save):std::string("N/A (<=0)"))
             <<" steps\n\n";
    return 0;
}
