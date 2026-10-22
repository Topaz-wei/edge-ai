#include "edge/video/double_buffer.hpp"
#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
using edge::video::DoubleBuffer;
void require(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
int main() {
    try {
        std::atomic<int> produced{0};
        std::promise<void> second;
        auto ready=second.get_future();
        DoubleBuffer<int> buffer([&](int& value,std::size_t index) {
            if(index==5)return false;
            value=static_cast<int>(index);
            ++produced;
            if(index==1)second.set_value();
            return true;
        });
        auto first=buffer.acquire();
        require(first && *first==0,"First slot");
        require(ready.wait_for(std::chrono::seconds(2))==std::future_status::ready,"No overlapping producer");
        require(produced==2 && *first==0,"Live slot overwritten or unbounded producer");
        buffer.release();
        for(int i=1;i<5;++i) {
            auto value=buffer.acquire();require(value && *value==i,"Out-of-order slot");buffer.release();
        }
        require(!buffer.acquire(),"EOF lost");
        DoubleBuffer<int> failing([](int& value,std::size_t index) {
            if(index==2)throw std::runtime_error("producer failure");
            value=index;return true;
        });
        for(int i=0;i<2;++i) {
            auto value=failing.acquire();require(value && *value==i,"Error overtook ready frame");failing.release();
        }
        bool rejected=false;
        try{failing.acquire();}catch(const std::runtime_error& e){rejected=std::string(e.what())=="producer failure";}
        require(rejected,"Producer exception lost");
        // Destruction must wake a producer waiting for space, including exception unwinding.
        try {
            DoubleBuffer<int> cancelled([](int& value,std::size_t i){value=i;return true;});
            require(cancelled.acquire()!=nullptr,"Cancellation fixture empty");
            throw std::runtime_error("consumer failure");
        }catch(const std::runtime_error&){}
        DoubleBuffer<int> empty([](int&,std::size_t){return false;});
        require(!empty.acquire(),"Empty stream");
        std::cout<<"Double buffer ordering, overlap, ownership, errors and cancellation passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
