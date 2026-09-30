/* SPDX-License-Identifier: MIT */
#include "nir_interpreter.h"
#include "nir_builder.h"
#include "nir_serialize.h"
#include "util/blob.h"
#include "sid.h"
#include <fstream>
#include <iostream>

struct params {
   uint64_t source, output, stream, count;
   uint32_t sequences, stride, code, records;
};
static std::vector<uint8_t> read(const std::string &p)
{
   std::ifstream f(p,std::ios::binary);
   if (!f) throw std::runtime_error("missing input " + p);
   return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),{});
}
static uint32_t get(const std::vector<uint8_t> &v, size_t off)
{
   uint32_t x; if(off+4>v.size())throw std::runtime_error("read outside output");
   memcpy(&x,v.data()+off,4);return x;
}
static void set(std::vector<uint8_t> &v,size_t off,uint32_t x) { memcpy(v.data()+off,&x,4); }
static void check(bool b,const char *why) { if(!b)throw std::runtime_error(why); }
static void pm4(const std::vector<uint8_t> &v,size_t begin,size_t bytes,bool task,unsigned draws)
{
   unsigned dispatch=0,mesh=0,flush=0,native=0;
   bool producer=false,producer_flush=false,producer_acquire=false;
   for(size_t off=begin;off<begin+bytes;) {
      uint32_t h=get(v,off);check((h>>30)==3 || (h>>30)==2,"invalid PM4 header");
      unsigned len=(h==PKT3_NOP_PAD || (h>>30)==2) ? 1 : ((h>>16)&0x3fff)+2;
      check(off+len*4<=begin+bytes,"packet crosses program boundary");
      if(task)check((off-begin)%1048576+len*4<=1048576,"packet crosses Task IB boundary");
      unsigned op=(h>>8)&255;
      if(task && op==PKT3_COND_EXEC)
         check((off-begin)%1048576+(len+get(v,off+(len-1)*4))*4<=1048576,"conditional execution crosses Task IB boundary");
      if(op==PKT3_DISPATCH_INDIRECT || op==PKT3_DISPATCH_DIRECT) {
         dispatch++;producer=true;producer_flush=false;producer_acquire=false;
      }
      if(op==PKT3_EVENT_WRITE && len>1 && (get(v,off+4)&63)==V_028A90_CS_PARTIAL_FLUSH && producer)
         producer_flush=true;
      if(op==PKT3_ACQUIRE_MEM && producer_flush)producer_acquire=true;
      if(op==PKT3_DISPATCH_MESH_INDIRECT_MULTI) {
         mesh++;
         if(task)check(producer && producer_flush && producer_acquire,"consumer precedes Task producer visibility barrier");
         producer=false;producer_flush=false;producer_acquire=false;
      }
      if(op==PKT3_EVENT_WRITE)flush++;
      if(op==PKT3_DISPATCH_TASKMESH_GFX || op==PKT3_DISPATCH_TASKMESH_DIRECT_ACE || op==PKT3_DISPATCH_TASKMESH_INDIRECT_MULTI_ACE)native++;
      off+=len*4;
   }
   check(!native,"native Task packet on hybrid Task route");
   check(mesh>0,"missing Mesh consumer");
   if(task) {
      check(dispatch>=1025*draws,"missing setup/Task producer dispatches");
      check(mesh==1024*draws,"missing per-chunk Mesh consumers");
      check(flush>=1024*draws,"missing producer/consumer flushes");
   }
}
static void task_count(const std::vector<uint8_t> &v,const params &p,unsigned seq,const std::string &dir)
{
   size_t base=size_t(seq)*p.stride, templ=p.code-1048576;
   check(templ==size_t(p.records)*131072,"incorrect reusable Task shape");
   auto sites=read(dir+"/task-uploads-"+std::to_string(seq)+".bin");
   check(sites.size() && sites.size()%4==0 && sites.size()/4<=2050,"invalid typed Task upload list");
   for(unsigned draw=0;draw<p.records;draw++) {
      unsigned writes=0,calls=0;bool drain=false,acquire=false,sync=false;
      for(size_t off=base+size_t(draw)*131072;off<base+size_t(draw+1)*131072;) {
         uint32_t h=get(v,off);unsigned len=(h==PKT3_NOP_PAD || h>>30==2)?1:((h>>16)&0x3fff)+2;
         check(off+len*4<=base+size_t(draw+1)*131072,"patch crosses IB boundary");unsigned op=h>>8&255;
         if(op==PKT3_EVENT_WRITE && (get(v,off+4)&63)==V_028A90_CS_PARTIAL_FLUSH)drain=true;
         if(op==PKT3_WRITE_DATA) {
            check(drain && len==9,"constant write lacks drain or wrong size");
            check(writes<sites.size()/4,"unexpected private write");
            uint64_t va=uint64_t(get(v,off+8))|(uint64_t(get(v,off+12))<<32);
            unsigned offset=get(sites,writes*4);
            check(va==p.output+base+p.code+offset+20,"patch writes outside typed constants");
            check(get(v,off+16)==draw,"patched DrawID differs");
            check(get(v,off+20)==get(v,base+p.code+offset+24) && get(v,off+24)==get(v,base+p.code+offset+28),"application constants pointer changed");
            uint64_t input=uint64_t(get(v,off+28))|(uint64_t(get(v,off+32))<<32);
            check(input==p.output+base+p.code+4+draw*12,"patched input differs");writes++;
         }
         if(op==PKT3_ACQUIRE_MEM && writes==sites.size()/4)acquire=true;
         if(op==PKT3_PFP_SYNC_ME && acquire)sync=true;
         if(op==PKT3_INDIRECT_BUFFER) {
            check(writes==sites.size()/4 && drain && acquire && sync,"template call precedes publication");
            uint64_t va=uint64_t(get(v,off+4))|(uint64_t(get(v,off+8))<<32);
            check(va==p.output+base+templ && (get(v,off+12)&0xfffff)==1048576/4,"wrong template IB");calls++;
            check(get(v,off+12)&S_3F3_CHAIN(1),"template nests an unsupported IB3");
         }
         off+=len*4;
      }
      check(calls==1 && writes==sites.size()/4,"incomplete Task patch/call");
   }
   pm4(v,base+templ,1048576,true,1);
}
int main(int argc,char **argv)
{
   try {
      check(argc==4 || argc==5,"usage: oracle dump-directory count-token task-route [push-constants]");
      bool pcs=argc==5 && atoi(argv[4]);
      std::string dir=argv[1];bool count=atoi(argv[2]),task=atoi(argv[3]);
      std::string suffix=std::to_string(count);
      auto capture=read(dir+"/capture-"+suffix+".bin");
      check(capture.size()>=sizeof(params),"truncated capture");
      params p;memcpy(&p,capture.data(),sizeof(p));
      std::vector<uint8_t> src(capture.begin()+sizeof(p),capture.end());
      check(src.size()==uint64_t(p.stride)*p.sequences,"incorrect snapshot bounds");
      auto token=read(dir+"/token-"+suffix+".bin");auto blob=read(dir+"/prepare-"+suffix+".nir");
      struct blob_reader reader;blob_reader_init(&reader,blob.data(),blob.size());
      nir_shader_compiler_options options={};
      nir_shader *s=nir_deserialize(nullptr,&options,&reader);
      check(s && !reader.overrun,"invalid NIR capture");
      nir_lower_vars_to_ssa(s);nir_opt_dce(s);nir_index_ssa_defs(nir_shader_get_entrypoint(s));
      nir_validate_shader(s,"DGC CPU oracle input");
      for(unsigned seq_count : {0u,1u,p.sequences,p.sequences+7}) {
         set(token,p.count-p.stream,seq_count);
         for(unsigned draw_count : {0u,1u,p.records,p.records+7}) {
            if(count){set(token,12,draw_count);set(token,44,draw_count);}
            std::vector<uint8_t> dst(src.size()+256,0xA5);
            bc250_dgc_nir_interpreter interp(s);
            interp.push_constants.resize(sizeof(p));memcpy(interp.push_constants.data(),&p,sizeof(p));
            interp.mappings={{p.source,src.size(),src.data()},{p.output,src.size(),dst.data()},{p.stream,token.size(),token.data()}};
            for(unsigned seq=0;seq<p.sequences;seq++)for(unsigned lane=0;lane<64;lane++)interp.run(s,seq,lane);
            for(unsigned seq=0;seq<p.sequences;seq++) {
               size_t base=size_t(seq)*p.stride;bool active=seq<std::min(seq_count,p.sequences);
               if(active) {
                  check(!memcmp(dst.data()+base,src.data()+base,p.code),"generated PM4 differs from ordinary capture");
                  if(task && count)task_count(dst,p,seq,dir);
                  else pm4(dst,base,p.code,task,1);
               } else for(size_t i=0;i<p.code;i+=4)check(get(dst,base+i)==PKT3_NOP_PAD,"inactive sequence executes commands");
               unsigned n=count ? std::min(draw_count,p.records) : seq+1;
               check(get(dst,base+p.code)==(active?n:0),"count or sequence clamp differs");
               for(unsigned record=0;record<p.records;record++)for(unsigned c=0;c<3;c++) {
                  bool valid=active && (count ? record<n : record==seq);
                  uint32_t expected=valid ? get(token,(count?64+record*16:seq*32)+c*4):0;
                  check(get(dst,base+p.code+4+record*12+c*4)==expected,"DrawID/grid record differs");
               }
               size_t skip=4+p.records*12;
               if(pcs) {
                  size_t app=(skip+15)&~15u;
                  for(unsigned word=0;word<64;word++) {
                     uint32_t expected=word==0 ? get(token,seq*32+16) : word==1 ? seq : get(src,base+p.code+app+word*4);
                     check(get(dst,base+p.code+app+word*4)==expected,"push constant/sequence index differs");
                  }
                  skip=app+256;
               }
               check(!memcmp(dst.data()+base+p.code+skip,src.data()+base+p.code+skip,p.stride-p.code-skip),"private upload changed");
            }
            for(size_t i=src.size();i<dst.size();i++)check(dst[i]==0xA5,"preprocess output overrun");
         }
      }
      ralloc_free(s);
      std::cout << "DGC_CPU_PM4_PASS count=" << count << " task=" << task << " cases=16\n";
      return 0;
   } catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}
}
