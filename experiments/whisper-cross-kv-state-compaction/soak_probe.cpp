// Stability probe: does the cross-round heap corruption depend on request
// teardown, or on batch-shrink set_state itself? Two modes:
//   fresh : new InferRequest each round (matches production cohort loop)
//   reuse : one InferRequest, reset_state() between rounds
// Each round: prefill B=4, one shrink to 3 survivors via cross-KV set_state, a
// few decode steps. Report how many rounds complete before crash.
#include <openvino/openvino.hpp>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>
#include <string>
int main(int argc,char** argv){
    std::string dec=argv[1]; std::string mode=argc>2?argv[2]:"fresh"; int ROUNDS=argc>3?std::stoi(argv[3]):10;
    size_t E=1500,D=384,B=4,P=4;
    std::vector<int64_t> PROMPT={50258,50259,50359,50363};
    ov::Core core; auto cm=core.compile_model(core.read_model(dec),"CPU");
    std::vector<float> enc((size_t)B*E*D,0.1f);
    auto set_all=[&](ov::InferRequest&r,size_t W,size_t seqlen,int64_t fill){
        ov::Tensor e(ov::element::f32,ov::Shape{W,E,D}); std::memcpy(e.data<float>(),enc.data(),(size_t)W*E*D*4);
        ov::Tensor ids(ov::element::i64,ov::Shape{W,seqlen});
        for(size_t i=0;i<W*seqlen;++i) ids.data<int64_t>()[i]= (seqlen==P?PROMPT[i%P]:fill);
        ov::Tensor bi(ov::element::i32,ov::Shape{W}); for(size_t i=0;i<W;++i)bi.data<int32_t>()[i]=(int32_t)i;
        r.set_tensor("encoder_hidden_states",e); r.set_tensor("input_ids",ids); r.set_tensor("beam_idx",bi);
    };
    ov::InferRequest reuse; if(mode=="reuse") reuse=cm.create_infer_request();
    int done=0;
    for(int rd=0;rd<ROUNDS;++rd){
        ov::InferRequest r = (mode=="reuse")? reuse : cm.create_infer_request();
        if(mode=="reuse") r.reset_state();
        set_all(r,B,P,0); r.infer();                        // prefill B=4
        // one shrink to survivors {0,1,2}
        std::vector<int32_t> surv={0,1,2};
        auto sts=r.query_state();
        for(auto&st:sts){ ov::Tensor cur=st.get_state(); auto sh=cur.get_shape(); if(sh[2]!=E)continue;
            size_t re=sh[1]*sh[2]*sh[3]; ov::Tensor comp(ov::element::f32,ov::Shape{surv.size(),sh[1],sh[2],sh[3]});
            const float* s=cur.data<float>(); float* d=comp.data<float>();
            for(size_t k=0;k<surv.size();++k) std::memcpy(d+k*re,s+(size_t)surv[k]*re,re*4);
            st.set_state(comp); }
        sts.clear();
        // decode 3 steps at width 3
        { ov::Tensor e(ov::element::f32,ov::Shape{3,E,D}); std::memcpy(e.data<float>(),enc.data(),(size_t)3*E*D*4);
          ov::Tensor ids(ov::element::i64,ov::Shape{3,1}); for(int i=0;i<3;++i)ids.data<int64_t>()[i]=100;
          ov::Tensor bi(ov::element::i32,ov::Shape{3}); bi.data<int32_t>()[0]=0;bi.data<int32_t>()[1]=1;bi.data<int32_t>()[2]=2;
          r.set_tensor("encoder_hidden_states",e); r.set_tensor("input_ids",ids); r.set_tensor("beam_idx",bi); r.infer(); }
        done=rd+1;
        std::cout<<"round "<<done<<" ok"<<std::endl;
    }
    std::cout<<"MODE="<<mode<<" completed "<<done<<"/"<<ROUNDS<<" rounds"<<std::endl;
    return 0;
}
