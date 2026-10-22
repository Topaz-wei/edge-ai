#pragma once
#include <array>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

namespace edge::video {
// One producer, one consumer, exactly two live slots. The consumer borrows the
// acquired slot until release(); the producer cannot reuse it before then.
// Producer errors and EOF are published in frame order. Destruction cancels
// waits and joins; it cannot interrupt an external blocking producer operation.
template<class T>
class DoubleBuffer {
public:
    using Produce=std::function<bool(T&,std::size_t)>;
    explicit DoubleBuffer(Produce produce):worker_([this,produce=std::move(produce)] {
        for(std::size_t index=0;;++index) {
            auto& slot=slots_[index%2];
            {
                std::unique_lock<std::mutex> lock(mutex_);
                changed_.wait(lock,[&]{return stopping_ || !slot.ready;});
                if(stopping_)return;
            }
            bool end=false;
            std::exception_ptr error;
            try { end=!produce(slot.value,index); }
            catch(...) { error=std::current_exception(); }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                slot.end=end;
                slot.error=error;
                slot.ready=true;
            }
            changed_.notify_all();
            if(end || error)return;
        }
    }){}
    DoubleBuffer(const DoubleBuffer&)=delete;
    DoubleBuffer& operator=(const DoubleBuffer&)=delete;
    ~DoubleBuffer() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_=true;
        }
        changed_.notify_all();
        worker_.join();
    }
    T* acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        auto& slot=slots_[next_%2];
        changed_.wait(lock,[&]{return slot.ready;});
        if(slot.error)std::rethrow_exception(slot.error);
        return slot.end?nullptr:&slot.value;
    }
    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& slot=slots_[next_%2];
            // Drop all references before allowing the producer to reuse storage.
            slot.value=T{};
            slot.ready=false;
            ++next_;
        }
        changed_.notify_all();
    }
private:
    struct Slot { T value{}; bool ready=false,end=false; std::exception_ptr error; };
    std::array<Slot,2> slots_;
    std::mutex mutex_;
    std::condition_variable changed_;
    bool stopping_=false;
    std::size_t next_=0;
    std::thread worker_; // Last: all shared state is initialized before launch.
};
}
