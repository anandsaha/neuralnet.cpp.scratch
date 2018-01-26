#include <iostream>
#include <string>
#include <cassert>

using namespace std;

// Configuration
// -----------------------------------------------------------------------------
const char* train_data  = "data/train-images-idx3-ubyte";
const char* train_label = "data/train-labels-idx1-ubyte";
const char* test_data   = "data/t10k-images-idx3-ubyte";
const char* test_label  = "data/t10k-labels-idx1-ubyte";

const size_t num_epochs = 100;
const size_t batch_size = 100;

// Error messages
// -----------------------------------------------------------------------------
string msg1 = "Tensor2D operation: indexes were out of range";
string msg2 = "left and right tensors do not have appropriate dimensions for dot product";

// Tensor infrastructure
// -----------------------------------------------------------------------------
template <typename T>
class Tensor2D
{
    public:
        explicit Tensor2D(size_t rows, size_t cols)
            : _rows(rows), _cols(cols){
            _data = new T*[_rows];
            for(size_t r = 0; r < _rows; r++)
                _data[r] = new T[_cols];
        }

        ~Tensor2D() {
            for(size_t r = 0; r < _rows; ++r)
                delete[] _data[r];
            delete[] _data;
        }

        T& get(size_t r, size_t c) { 
            if (r >= _rows || c >= _cols) throw out_of_range(msg1.c_str());
            return _data[r][c]; 
        }

        const T& get(size_t r, size_t c) const { 
            if (r >= _rows || c >= _cols) throw out_of_range(msg1.c_str());
            return _data[r][c]; 
        }

        size_t rows() const { return _rows; }
        size_t cols() const { return _cols; }

    private:
        size_t      _rows;
        size_t      _cols;
        T**         _data;
};

template<typename T>
Tensor2D<T> dot(const Tensor2D<T>& left, const Tensor2D<T>& right)
{
    if(left.cols() != right.rows()) throw out_of_range(msg2.c_str());

    Tensor2D<T> t(left.rows(), right.cols());

    for(size_t r = 0; r < left.rows(); r++)
        for(size_t i = 0; i < left.cols(); i++)
            for(size_t c = 0; c < right.cols(); c++)
                t.get(r, c) += left.get(r, i) * right.get(i, c);

    return t;
}

// Reading train/test data
// -----------------------------------------------------------------------------
class MNISTDataLoader
{
    public:
        explicit MNISTDataLoader(const char* data_path, const char* label_path): 
            data_path(data_path), label_path(label_path) {
        }

        ~MNISTDataLoader() {}

    private:
        string data_path;
        string label_path;
};


// Neural Network
// -----------------------------------------------------------------------------

// Loss function
// -----------------------------------------------------------------------------

// Optimizer
// -----------------------------------------------------------------------------

template <typename T>
void p(const Tensor2D<T>& t)
{
    for(size_t i = 0; i < t.rows(); i++) {
        for(size_t j = 0; j < t.cols(); j++)
            cout << t.get(i, j) << " ";
        cout << endl;
    }

}

// -----------------------------------------------------------------------------
int main()
{
    Tensor2D<float> t1(5, 5);
    cout << t1.rows() << " " << t1.cols() << endl;

    t1.get(3, 3) = 1.2;
    t1.get(3, 4) = 2.5;

    p(t1);

    // t1.set(100, 100, 100);
    // cout << t1.get(100, 100);

    //Tensor2D<float> t2(51, 10);
    //dot(t1, t2);
    
    cout << "====" << endl;

    Tensor2D<float> l(2, 2);
    Tensor2D<float> r(2, 3);

    l.get(0, 0) = 1;
    l.get(0, 1) = 2;
    l.get(1, 0) = 3;
    l.get(1, 1) = 4;

    r.get(0, 0) = 5;
    r.get(0, 1) = 6;
    r.get(0, 2) = 7;
    r.get(1, 0) = 8;
    r.get(1, 1) = 9;
    r.get(1, 2) = 10;

    Tensor2D<float> result = dot(l, r);
    p(l);
    p(r);
    p(result);
    return 0;
}
