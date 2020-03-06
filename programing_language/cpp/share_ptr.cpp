#include <iostream>
#include <memory>

using namespace std;

class A {
public:
    int data;

    A(int n) : data(n) {};

    ~A() { cout << data << " " << "destructed" << endl; }
};

shared_ptr<A> get_ptr() {
    shared_ptr<A> ret = make_shared<A>(123);
    return ret;
}

void shared_ptr_test() {
    shared_ptr<A> sp1(new A(2));
    shared_ptr<A> sp2 = sp1;
    shared_ptr<A> sp3;
    sp3 = sp2;
    cout << sp1->data << "," << sp2->data << "," << sp3->data << endl;
    A *p = sp3.get();      // get返回托管的指针，p 指向 A(2)
    cout << p->data << endl;  //输出 2
    sp1 = std::make_shared<A>(3);    // reset导致托管新的指针, 此时sp1托管A(3)
    sp2 = std::make_shared<A>(4);    // sp2托管A(4)
    cout << sp1->data << endl; //输出 3
    sp3 = std::make_shared<A>(5);    // sp3托管A(5),A(2)无人托管，被delete
    cout << "end" << endl;

    shared_ptr<A> ret = get_ptr();
    cout << ret->data << endl;

}