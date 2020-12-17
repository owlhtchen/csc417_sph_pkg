#include <atomic>

class AtomicInt: public std::atomic<int> {
    // using std::atomic<int>::atomic;
    public:
    AtomicInt():AtomicInt(0){}
    AtomicInt(int data): std::atomic<int>::atomic(data) { };
    AtomicInt(const AtomicInt& rhs): std::atomic<int>::atomic(rhs.load()){};
    inline AtomicInt& operator=(const AtomicInt& other) // copy assignment
    {
        if (this != &other) { // self-assignment check expected
            std::atomic<int>::operator=(other.load());
        }
        return *this;
    }
};