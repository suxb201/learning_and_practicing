//
// Created by 苏晓斌 on 2020/3/6.
//

#include <iostream>
#include <memory>
#include <vector>

using namespace std;

class A {
public:
    int data;

    A(int n) : data(n) {};

    ~A() { cout << data << " " << "destructed" << endl; }
};

vector<A> return_vector() {
    vector<A> ret;

    ret.emplace_back(A(123));

    return ret;
}

void return_vector_test() {
    vector<A> ret = return_vector();
    cout << ret.front().data << endl;
    cout << "end" << endl;
}