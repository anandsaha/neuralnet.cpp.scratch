#include "mnist.h"

// Right now the tests are validated by visual inspection
// Can be binded to a unit testing framework for automation

pair<Tensor2D<precision>, Tensor2D<precision>>
getmock() {
    Tensor2D<precision> t1(2, 3);
    Tensor2D<precision> t2(3, 2);

    t1[0][0] = 1;
    t1[0][1] = 2;
    t1[0][2] = 3;
    t1[1][0] = 4;
    t1[1][1] = 5;
    t1[1][2] = 6;
    
    t2[0][0] = 9;
    t2[0][1] = 10;
    t2[1][0] = 11;
    t2[1][1] = 12;
    t2[2][0] = 13;
    t2[2][1] = 15;

    return pair<Tensor2D<precision>, Tensor2D<precision>>(t1, t2);
}

pair<Tensor2D<precision>, Tensor2D<precision>>
getmock2() {
    Tensor2D<precision> t1(2, 2);
    Tensor2D<precision> t2(2, 2);

    t1[0][0] = 1;
    t1[0][1] = 2;
    t1[1][0] = 3;
    t1[1][1] = 4;
 
    t2[0][0] = 5;
    t2[0][1] = 6;
    t2[1][0] = 7;
    t2[1][1] = 8;

    return pair<Tensor2D<precision>, Tensor2D<precision>>(t1, t2);
}


void test_tensor() {
    cout << "test_tensor" << endl;
    Tensor2D<precision> t(2, 3);
    t[0][0] = 1.0;
    pt(t);
}

void test_dot() {
    cout << "test_dot" << endl;
    auto p = getmock();
    pt(dot(p.first, p.second));
}

void test_add() {
    cout << "test_add" << endl;
    auto p = getmock2();
    pt(add(p.first, p.second));
}

void test_sub() {
    cout << "test_sub" << endl;
    auto p = getmock2();
    pt(sub(p.first, p.second));
}

void test_mul() {
    cout << "test_mul" << endl;
    auto p = getmock2();
    pt(mul(p.first, 1.1));
}

void test_transpose() {
    cout << "test_transpose" << endl;
    auto p = getmock();
    pt(transpose(p.first));
}

int main()
{
    test_tensor();
    test_dot();
    test_add();
    test_sub();
    test_mul();
    test_transpose();
    return 0;
}
