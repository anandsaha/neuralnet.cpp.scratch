#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <random>
#include <string>
#include <cstring>

using namespace std;

// Configuration
// -----------------------------------------------------------------------------
const char* train_data  = "data/train-images-idx3-ubyte";
const char* train_label = "data/train-labels-idx1-ubyte";
const char* test_data   = "data/t10k-images-idx3-ubyte";
const char* test_label  = "data/t10k-labels-idx1-ubyte";

const size_t num_epochs = 100;
const size_t batch_size = 100;
const size_t pixels = 784;

//typedef double precision;
typedef float precision;

// Error messages
// -----------------------------------------------------------------------------
string msg1 = "Tensor2D operation: indexes were out of range";
string msg2 = "left and right tensors do not have appropriate dimensions for dot product";
string msg3 = "could not open file for reading";

// Tensor infrastructure
// -----------------------------------------------------------------------------
template <typename T>
class Tensor2D
{
    public:
        explicit Tensor2D(size_t rows, size_t cols)
            : _rows(rows), _cols(cols) {
            _data = new T*[_rows];
            for(size_t r = 0; r < _rows; ++r)
                _data[r] = new T[_cols];
        }

        Tensor2D(const Tensor2D& rhs) {
            _rows = rhs._rows;
            _cols = rhs._cols;
            _data = new T*[_rows];
            for(size_t r = 0; r < _rows; ++r)
                _data[r] = new T[_cols];
            for(size_t r = 0; r < _rows; ++r)
                for(size_t c = 0; c < _cols; ++c)
                    _data[r][c] = rhs._data[r][c];
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

        T* getrow(size_t r) {
            if (r >= _rows) throw out_of_range(msg1.c_str());
            return _data[r];
        }

        size_t rows() const { return _rows; }
        size_t cols() const { return _cols; }

    private:
        size_t      _rows;
        size_t      _cols;
        T**         _data;
        Tensor2D& operator=(const Tensor2D&);
};


// Dot product operation
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

// Reading train/test data with iterator
// -----------------------------------------------------------------------------

int b2i(const char* ptr, size_t idx) {
    int val = 0;
    val |= (unsigned char)ptr[idx+0]; val <<= 8;
    val |= (unsigned char)ptr[idx+1]; val <<= 8;
    val |= (unsigned char)ptr[idx+2]; val <<= 8;
    val |= (unsigned char)ptr[idx+3];
    return val;
}

typedef pair<Tensor2D<precision>, Tensor2D<int> > batchtype;

class MNISTDataLoader
{
    public:
        explicit MNISTDataLoader(const char* data_path, const char* label_path)
            : _data(NULL), _label(NULL) {
            
            _data_size  = fill(_data, data_path);
            _label_size = fill(_label, label_path);
            _num_items  = b2i(_data, 4);
            _num_rows   = b2i(_data, 8);
            _num_cols   = b2i(_data, 12);
            assert(_num_items == b2i(_label, 4));
        }

        ~MNISTDataLoader() {
            delete[] _data;
            delete[] _label;
        }

        batchtype fetch(int batch_size) {

            Tensor2D<precision> data(batch_size, pixels);
            Tensor2D<int>       label(batch_size, 1);

            // TODO - put this in constructor
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, _num_items-1);

            for(int i = 0; i < batch_size; ++i) {
                size_t offset = dis(gen);
                label.get(i, 0) = _label[8 + offset];
                // TODO - optimize
                int off = 16 + (offset * pixels);
                for(int p = 0; p < pixels; p++) 
                    data.get(i, p) = (int)((unsigned char)_data[off + p]);
            }

            return batchtype(data, label);
        }

    private:
        char* _data, * _label;
        size_t _data_size, _label_size, _num_items, _num_rows, _num_cols;

        /*
        I am assuming that we have enough RAM to load the entire 
        dataset into memory at once (~54 MB for MNIST). If not, I 
        would have loaded it part by part on the fly.
        */
        size_t fill(char*& target, const char* file_path) {
            ifstream fd(file_path, ios::in|ios::binary);
            if(fd) {
                fd.seekg(0, ios::end);
                std::fstream::pos_type size = fd.tellg();
                fd.seekg(0, ios::beg);
                target = new char[size];
                fd.read(&target[0], size);
                fd.close();
                return size;
            } else {
                throw runtime_error(msg3.c_str());
            }
        }
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
    Tensor2D<precision> t1(5, 5);
    cout << t1.rows() << " " << t1.cols() << endl;

    t1.get(3, 3) = 1.2;
    t1.get(3, 4) = 2.5;

    p(t1);

    // t1.set(100, 100, 100);
    // cout << t1.get(100, 100);

    //Tensor2D<float> t2(51, 10);
    //dot(t1, t2);
    
    cout << "====" << endl;



    Tensor2D<precision> l(2, 2);
    Tensor2D<precision> r(2, 3);
    Tensor2D<precision> b(4, 3);

    //dot(l, b);

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

    Tensor2D<precision> result = dot(l, r);
    p(l);
    p(r);
    p(result);


    MNISTDataLoader train(train_data, train_label);
    MNISTDataLoader test(test_data, test_label);

    batchtype p = train.fetch(2);

    for(int x = 0; x < 2; x++) {
        cout << p.second.get(x, 0) << endl;
        cout << "[";
        for(int i = 0; i < 28; i++) {
            cout << "[";
            for(int j = 0; j < 28; j++)
                cout << p.first.get(x, (28*i) + j) << ",";
            cout << "], ";
        }
        cout << "]" << endl;;
    }



    return 0;
}
