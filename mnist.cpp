#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <random>
#include <string>
#include <cstring>
#include <cmath>

using namespace std;

// Configuration
// -----------------------------------------------------------------------------
const char* train_data  = "data/train-images-idx3-ubyte";
const char* train_label = "data/train-labels-idx1-ubyte";
const char* test_data   = "data/t10k-images-idx3-ubyte";
const char* test_label  = "data/t10k-labels-idx1-ubyte";

const size_t num_epochs = 5;
const size_t batch_size = 100;
const size_t pixels     = 784; // 28 * 28

typedef float precision;       // what precision to use for the tensors

// Get random numbers sampled from normal distribution, for initializing weights
default_random_engine generator;
normal_distribution<precision> distribution(0.0, 0.01);
precision genrand() {
    return distribution(generator);
}

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

            fill(0.0);
        }

        Tensor2D(const Tensor2D& rhs) {
            copy(rhs);
        }

        void assign(const Tensor2D& rhs) {
            if(this != &rhs) {
                freemem();
                copy(rhs);
            }
        }

        ~Tensor2D() {
            freemem();
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
        
        void fill(T setval) {
            for(size_t r = 0; r < _rows; ++r)
                for(size_t c = 0; c < _cols; ++c)
                    _data[r][c] = setval;
        }

        size_t rows() const { return _rows; }
        size_t cols() const { return _cols; }

    private:
        size_t      _rows;
        size_t      _cols;
        T**         _data;

        void copy(const Tensor2D& rhs) {
            _rows = rhs._rows;
            _cols = rhs._cols;
            _data = new T*[_rows];
            for(size_t r = 0; r < _rows; ++r)
                _data[r] = new T[_cols];
            for(size_t r = 0; r < _rows; ++r)
                for(size_t c = 0; c < _cols; ++c)
                    _data[r][c] = rhs._data[r][c];
        }

        void freemem() {
            for(size_t r = 0; r < _rows; ++r)
                delete[] _data[r];
            delete[] _data;
        }

        // Clients should use assign() function
        Tensor2D& operator=(const Tensor2D& rhs);
};

// TODO - Delete me
template <typename T>
void px(const Tensor2D<T>& t)
{
    for(size_t i = 0; i < t.rows(); i++) {
        for(size_t j = 0; j < t.cols(); j++)
            cout << t.get(i, j) << " ";
        cout << endl;
    }

}

// Dot product between Tensor2D objects
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

// Add Tensor2D objects, with broadcasting
template<typename T>
Tensor2D<T> add(const Tensor2D<T>& left, const Tensor2D<T>& right)
{
    assert(left.cols() == right.cols());
    assert( (left.rows() == right.rows()) || (right.rows() == 1));
    Tensor2D<T> t(left.rows(), left.cols());

    for(size_t r = 0; r < left.rows(); ++r)
        for(size_t c = 0; c < left.cols(); ++c) {
            if(right.rows() > 1)
                t.get(r, c) = left.get(r, c) + right.get(r, c);
            else
                t.get(r, c) = left.get(r, c) + right.get(0, c);
        }

    return t;
}

// Sub Tensor2D objects, with broadcasting
template<typename T>
Tensor2D<T> sub(const Tensor2D<T>& left, const Tensor2D<T>& right)
{
    assert(left.cols() == right.cols());
    assert( (left.rows() == right.rows()) || (right.rows() == 1));
    Tensor2D<T> t(left.rows(), left.cols());

    for(size_t r = 0; r < left.rows(); ++r)
        for(size_t c = 0; c < left.cols(); ++c) {
            if(right.rows() > 1)
                t.get(r, c) = left.get(r, c) - right.get(r, c);
            else
                t.get(r, c) = left.get(r, c) - right.get(0, c);
        }

    return t;
}

// Mul Tensor2D objects
template<typename T>
Tensor2D<T> mul(const Tensor2D<T>& left, float x)
{
    Tensor2D<T> t(left.rows(), left.cols());

    for(size_t r = 0; r < left.rows(); ++r)
        for(size_t c = 0; c < left.cols(); ++c)
            t.get(r, c) = left.get(r, c) * x;

    return t;
}


// Transpose a Tensor2D object
template<typename T>
Tensor2D<T> transpose(const Tensor2D<T>& input) {
    Tensor2D<T> t(input.cols(), input.rows());

    for(size_t r = 0; r < input.rows(); ++r)
        for(size_t c = 0; c < input.cols(); ++c)
            t.get(c, r) = input.get(r, c);

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

// -----------------------------------------------------------------------------
// Section: Loss function
// -----------------------------------------------------------------------------
template <typename T>
T maxval(T* data, size_t count) {
    T m = 0;
    for(size_t i = 0; i < count; ++i)
        if (data[i] > m) m = data[i];
    return m;
}

template <typename T>
size_t maxidx(T* data, size_t count) {
    T m = 0;
    size_t idx = -1;
    for(size_t i = 0; i < count; ++i)
        if (data[i] > m) { m = data[i]; idx = i; } 
    return idx;
}


// Numerically stable softmax
template <typename T>
Tensor2D<T> softmax(Tensor2D<T> scores) {

    Tensor2D<T> max(scores.rows(), 1);
    Tensor2D<T> expsum(scores.rows(), 1);

    for(size_t r = 0; r < scores.rows(); ++r)
        max.get(r, 0) = maxval(scores.getrow(r), scores.cols());

    for(size_t r = 0; r < scores.rows(); ++r)
        for(size_t c = 0; c < scores.cols(); ++c)
            scores.get(r, c) = scores.get(r, c) - max.get(r, 0);

    for(size_t r = 0; r < scores.rows(); ++r)
        for(size_t c = 0; c < scores.cols(); ++c)
            expsum.get(r, 0) += exp(scores.get(r, c)); 

    for(size_t r = 0; r < scores.rows(); ++r)
        for(size_t c = 0; c < scores.cols(); ++c)
            scores.get(r, c) = exp(scores.get(r, c)) / expsum.get(r, 0);

    return scores;
}

template<typename T>
T logloss(const Tensor2D<int>& actual, const Tensor2D<T>& prediction) {
    T loss = 0.0;
    assert(actual.rows() == prediction.rows());
    for(size_t r = 0; r < actual.rows(); ++r) {
        size_t idx = actual.get(r, 0);
        loss += -1.0 * log(prediction.get(r, idx));
    }
    return loss / actual.rows();
}

// -----------------------------------------------------------------------------
// Section: Neural Network
// -----------------------------------------------------------------------------

// ReLU activation function
template <typename T>
Tensor2D<T>& relu(Tensor2D<T>& input) {
    for(size_t r = 0; r < input.rows(); r++)
        for(size_t c = 0; c < input.cols(); c++)
            if(input.get(r, c) <= 0.0)
                input.get(r, c) = 0.0;
    return input;
}

// Linear layer
template <typename T>
class Linear 
{
    public:
        explicit Linear(size_t in, size_t out, bool add_relu = true)
            : weights(in, out), biases(1, out), activations(0, 0), 
              weights_grad(in, out), biases_grad(1, out), add_relu(add_relu) {
                init();
        }

        void forward(const Tensor2D<T>& input) {
            auto scores = add(dot(input, weights), biases);
            if(add_relu) activations.assign(relu(scores));
            else activations.assign(scores);
        }

        void backward(const Tensor2D<T>& grads) {
        }

        void clear() {
            weights_grad.fill(0.0);
            biases_grad.fill(0.0);
            activations.fill(0.0);
        }

        const Tensor2D<T>& getacts() const {
            return activations;
        }

    public:
        Tensor2D<T> weights;
        Tensor2D<T> biases;
        Tensor2D<T> weights_grad;
        Tensor2D<T> biases_grad;
        Tensor2D<T> activations;
        bool add_relu;

        void init() {
            for(size_t i = 0; i < weights.rows(); ++i)
                for(size_t j = 0; j < weights.cols(); ++j)
                    weights.get(i, j) = genrand();
            for(size_t i = 0; i < biases.rows(); ++i)
                for(size_t j = 0; j < biases.cols(); ++j)
                    biases.get(i, j) = 0.001;
        }
};

// The Network
template <typename T>
class Network
{
    public:
        explicit Network(size_t in, size_t out, size_t h1, size_t h2) 
            : layer1(in, h1), layer2(h1, h2), layer3(h2, out, false) {
        }

        Tensor2D<T> forward(const Tensor2D<T>& input) {

            layer1.forward(input);
            layer2.forward(layer1.getacts());
            layer3.forward(layer2.getacts());
            return softmax(layer3.getacts());
        }

        void backward(Tensor2D<T> sm, const Tensor2D<int>& actual, const Tensor2D<T>& input) {
            
            // Backprop through softmax
            for(size_t r = 0; r < sm.rows(); ++r)
                sm.get(r, actual.get(r, 0)) -= 1;

            for(size_t r = 0; r < sm.rows(); ++r)
                for(size_t c = 0; c < sm.cols(); ++c)
                    sm.get(r, c) /= sm.rows();

            // Backprop through layer3 
            layer3.weights_grad.assign(dot(transpose(layer2.getacts()), sm));
            for(size_t r = 0; r < sm.rows(); ++r)
                for(size_t c = 0; c < sm.cols(); ++c)
                    layer3.biases_grad.get(0, c) += sm.get(r, c);

            // Backprop through layer2 
            auto hidden2 = dot(sm, transpose(layer3.weights));
            for(size_t r = 0; r < hidden2.rows(); ++r)
                for(size_t c = 0; c < hidden2.rows(); ++c)
                    if (layer2.getacts().get(r, c) == 0)
                        hidden2.get(r, c) = 0.0;

            layer2.weights_grad.assign(dot(transpose(layer1.getacts()), hidden2));
            for(size_t r = 0; r < hidden2.rows(); ++r)
                for(size_t c = 0; c < hidden2.cols(); ++c)
                    layer2.biases_grad.get(0, c) += hidden2.get(r, c);

            // Backprop through layer1 
            auto hidden1 = dot(hidden2, transpose(layer2.weights));
            for(size_t r = 0; r < hidden1.rows(); ++r)
                for(size_t c = 0; c < hidden1.rows(); ++c)
                    if (layer1.getacts().get(r, c) == 0)
                        hidden1.get(r, c) = 0.0;

            layer1.weights_grad.assign(dot(transpose(input), hidden1));
            for(size_t r = 0; r < hidden1.rows(); ++r)
                for(size_t c = 0; c < hidden1.cols(); ++c)
                    layer1.biases_grad.get(0, c) += hidden1.get(r, c);


        }

        void opt(float lr=0.0001) {
            layer1.weights.assign(sub(layer1.weights, mul(layer1.weights_grad, lr)));
            layer2.weights.assign(sub(layer2.weights, mul(layer2.weights_grad, lr)));
            layer3.weights.assign(sub(layer3.weights, mul(layer3.weights_grad, lr)));

            layer1.biases.assign(sub(layer1.biases, mul(layer1.biases_grad, lr)));
            layer2.biases.assign(sub(layer2.biases, mul(layer2.biases_grad, lr)));
            layer3.biases.assign(sub(layer3.biases, mul(layer3.biases_grad, lr)));

            clear();
        }

        void clear() {
            layer1.clear();
            layer2.clear();
            layer3.clear();
        }


    private:
        Linear<T> layer1;
        Linear<T> layer2;
        Linear<T> layer3;
};
// -----------------------------------------------------------------------------
// Section: Optimizer
// -----------------------------------------------------------------------------

int main()
{
    /*
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

    batchtype p1 = train.fetch(2);
    for(int x = 0; x < 2; x++) {
        cout << p1.second.get(x, 0) << endl;
        cout << "[";
        for(int i = 0; i < 28; i++) {
            cout << "[";
            for(int j = 0; j < 28; j++)
                cout << p1.first.get(x, (28*i) + j) << ",";
            cout << "], ";
        }
        cout << "]" << endl;;
    }
    Linear<float> ll(2, 4);
    ll.print();
    Tensor2D<float> a1 = ll.forward(l);
    p(a1);
*/

/*
    Tensor2D<precision> r(2, 3);
    r.get(0, 0) = 5;
    r.get(0, 1) = 6;
    r.get(0, 2) = 7;
    r.get(1, 0) = 8;
    r.get(1, 1) = 9;
    r.get(1, 2) = 10;

    Tensor2D<precision> y(2, 1);
    y.get(0, 0) = 2;
    y.get(1, 0) = 2;

    Linear<float> ll(3, 4);
    ll.forward(r); 
    // p(ll.getacts());

    Network<precision> nt(3, 10, 3, 4);
    auto sm = nt.forward(r);
    cout << logloss(y, sm) << endl;
    nt.backward(sm, y, r); 
    nt.opt();

    int i = 100;

    while(i-- > 0) {
        auto sm1 = nt.forward(r);
        cout << logloss(y, sm1) << endl;
        nt.backward(sm1, y, r); 
        nt.opt();
    }
*/
    //nt.print();
/*
    Tensor2D<precision> y(2, 1);
    y.get(0, 0) = 2;
    y.get(1, 0) = 2;

    px(r);
    auto sm = softmax(r);
    px(sm);
    cout << logloss(y, softmax(r)) << endl;
*/


    MNISTDataLoader train(train_data, train_label);
    MNISTDataLoader test(test_data, test_label);
    Network<precision> nt(784, 10, 512, 1024);

    int i = num_epochs;
    while(i-- >= 0) 
    {
        batchtype batch = train.fetch(batch_size);
        auto sm = nt.forward(batch.first);
        cout << logloss(batch.second, sm) << endl;
        size_t totcorrect = 0;
        for(size_t r = 0; r < sm.rows(); ++r) {
            // cout << batch.second.get(r, 0) << ", " << maxidx(sm.getrow(r), sm.cols()) << endl;
            if(batch.second.get(r, 0) == maxidx(sm.getrow(r), sm.cols()))
                totcorrect++;
        }
        cout << totcorrect << "/" << sm.rows() << endl;
        nt.backward(sm, batch.second, batch.first); 
        nt.opt(0.001);
    }

    return 0;
}

