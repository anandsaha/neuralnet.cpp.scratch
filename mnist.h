#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <random>
#include <string>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iomanip>

using namespace std;

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
const char* train_data  = "data/train-images-idx3-ubyte";
const char* train_label = "data/train-labels-idx1-ubyte";
const char* test_data   = "data/t10k-images-idx3-ubyte";
const char* test_label  = "data/t10k-labels-idx1-ubyte";

const size_t num_epochs = 5;
const size_t batch_size = 1000;
const size_t pixels     = 784; // 28 * 28
const float  wt_reg     = 0.5;
const float  learn_rate = 0.001;

typedef float precision;       // what precision to use for the tensors

// Get random numbers sampled from normal distribution, for initializing weights
default_random_engine generator;
normal_distribution<precision> distribution(0.0, 0.01);
precision genrand() {
    return distribution(generator);
}
// To generate uniform random numbers
std::random_device rd;
std::mt19937 gen(rd());

string gettime() 
{
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%I:%M:%S",timeinfo);
    std::string str(buffer);
    return str;
}

// -----------------------------------------------------------------------------
// Error messages
// -----------------------------------------------------------------------------
string msg1 = "Tensor2D operation: indexes were out of range";
string msg2 = "left and right tensors do not have appropriate dimensions for dot product";
string msg3 = "could not open file for reading";

// -----------------------------------------------------------------------------
// Tensor infrastructure and operations
// -----------------------------------------------------------------------------

template <typename T>
class Tensor2D
{
    public:
        explicit Tensor2D(size_t rows, size_t cols)
            : _rows(rows), _cols(cols), _data(nullptr) {
                alloc(_rows, _cols);
                fill(0.0);
        }

        Tensor2D(const Tensor2D& rhs) {
            copy(rhs);
        }

        Tensor2D& operator=(const Tensor2D& rhs) {
            if(this != &rhs) {
                dealloc();
                copy(rhs);
            }
            return *this;
        }

        ~Tensor2D() {
            dealloc();
        }

        T* const operator[](size_t r) {
            return _data[r];
        }

        const T* const operator[](size_t r) const {
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

            alloc(_rows, _cols);

            for(size_t r = 0; r < _rows; ++r)
                for(size_t c = 0; c < _cols; ++c)
                    _data[r][c] = rhs._data[r][c];
        }

        void alloc(size_t rows, size_t cols) {
            _data = new T*[rows];
            for(size_t r = 0; r < rows; ++r)
                _data[r] = new T[cols];
        }

        void dealloc() {
            if (_data != nullptr) {
                for(size_t r = 0; r < _rows; ++r) {
                    if(_data[r] != nullptr) {
                        delete[] _data[r];
                        _data[r] = nullptr;
                    }
                }
                delete[] _data;
                _data = nullptr;
            }
            _rows = 0;
            _cols = 0;
        }

};

// Dot product between Tensor2D objects
template<typename T>
Tensor2D<T> dot(const Tensor2D<T>& left, const Tensor2D<T>& right)
{
    if(left.cols() != right.rows()) throw out_of_range(msg2.c_str());

    Tensor2D<T> t(left.rows(), right.cols());

    for(size_t r = 0; r < left.rows(); r++)
        for(size_t i = 0; i < left.cols(); i++)
            for(size_t c = 0; c < right.cols(); c++)
                t[r][c] += left[r][i] * right[i][c];

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
            if(right.rows() > 1) t[r][c] = left[r][c] + right[r][c];
            else t[r][c] = left[r][c] + right[0][c];
        }
    return t;
}

// Subtract Tensor2D objects, with broadcasting
template<typename T>
Tensor2D<T> sub(const Tensor2D<T>& left, const Tensor2D<T>& right)
{
    assert(left.cols() == right.cols());
    assert((left.rows() == right.rows()) || (right.rows() == 1));
    Tensor2D<T> t(left.rows(), left.cols());

    for(size_t r = 0; r < left.rows(); ++r)
        for(size_t c = 0; c < left.cols(); ++c) {
            if(right.rows() > 1) t[r][c] = left[r][c] - right[r][c];
            else t[r][c] = left[r][c] - right[0][c];
        }

    return t;
}

// Multiply a Tensor2D object with a scalar
template<typename T>
Tensor2D<T> mul(const Tensor2D<T>& left, float x)
{
    Tensor2D<T> t(left.rows(), left.cols());

    for(size_t r = 0; r < left.rows(); ++r)
        for(size_t c = 0; c < left.cols(); ++c)
            t[r][c] = left[r][c] * x;

    return t;
}


// Transpose a Tensor2D object
template<typename T>
Tensor2D<T> transpose(const Tensor2D<T>& input) {
    Tensor2D<T> t(input.cols(), input.rows());

    for(size_t r = 0; r < input.rows(); ++r)
        for(size_t c = 0; c < input.cols(); ++c)
            t[c][r] = input[r][c];
    return t;
}

// -----------------------------------------------------------------------------
// Reading MNIST train/test data with iterator
// -----------------------------------------------------------------------------

unsigned int b2i(const char* ptr, size_t idx) {
    unsigned int val = 0;
    val |= (unsigned char)ptr[idx+0]; val <<= 8;
    val |= (unsigned char)ptr[idx+1]; val <<= 8;
    val |= (unsigned char)ptr[idx+2]; val <<= 8;
    val |= (unsigned char)ptr[idx+3];
    return val;
}

// <data, label> pair
typedef pair<Tensor2D<precision>, Tensor2D<unsigned int> > batchtype;

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
            Tensor2D<unsigned int>       label(batch_size, 1);
            std::uniform_int_distribution<> dis(0, _num_items-1);

            for(int i = 0; i < batch_size; ++i) {
                size_t offset = dis(gen);
                label[i][0] = _label[8 + offset];
                int off = 16 + (offset * pixels);
                for(size_t p = 0; p < pixels; p++)
                    data[i][p] = (int)((unsigned char)_data[off + p]);
            }

            return batchtype(data, label);
        }

        size_t numitems() const {
            return _num_items;
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
// Loss function
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
        max[r][0] = maxval(scores[r], scores.cols());

    for(size_t r = 0; r < scores.rows(); ++r)
        for(size_t c = 0; c < scores.cols(); ++c)
            scores[r][c] = scores[r][c] - max[r][0];

    for(size_t r = 0; r < scores.rows(); ++r)
        for(size_t c = 0; c < scores.cols(); ++c)
            expsum[r][0] += exp(scores[r][c]); 

    for(size_t r = 0; r < scores.rows(); ++r)
        for(size_t c = 0; c < scores.cols(); ++c)
            scores[r][c] = exp(scores[r][c]) / expsum[r][0];

    return scores;
}

// Log loss cross entropy
template<typename T>
T logloss(const Tensor2D<unsigned int>& actual, const Tensor2D<T>& prediction) {
    T loss = 0.0;
    assert(actual.rows() == prediction.rows());
    for(size_t r = 0; r < actual.rows(); ++r) {
        size_t idx = actual[r][0];
        loss += -1.0 * log(prediction[r][idx]);
    }
    return loss / actual.rows();
}

// -----------------------------------------------------------------------------
// Neural Network
// -----------------------------------------------------------------------------

// ReLU activation function
template <typename T>
void relu(Tensor2D<T>& input) {
    for(size_t r = 0; r < input.rows(); r++)
        for(size_t c = 0; c < input.cols(); c++)
            if(input[r][c] <= 0.0)
                input[r][c] = 0.0;
}

// Linear layer
template <typename T>
class Linear 
{
    public:
        explicit Linear(size_t in, size_t out, bool add_relu = true)
            : weights(in, out), biases(1, out), weights_grad(in, out),
              biases_grad(1, out), activations(1, 1), add_relu(add_relu) {
                init();
        }

        void forward(const Tensor2D<T>& input) {
            auto scores = add(dot(input, weights), biases);
            if(add_relu) relu(scores);
            activations = scores;
        }

        Tensor2D<T> eval(const Tensor2D<T>& input) const {
            auto scores = add(dot(input, weights), biases);
            if(add_relu) relu(scores);
            return scores;
        }

        void clear() {
            weights_grad.fill(0.0);
            biases_grad.fill(0.0);
            activations.fill(0.0);
        }

        const Tensor2D<T>& getacts() const {
            return activations;
        }

        Tensor2D<T> weights;
        Tensor2D<T> biases;
        Tensor2D<T> weights_grad;
        Tensor2D<T> biases_grad;
        Tensor2D<T> activations;

    private:
        bool add_relu;

        void init() {
            for(size_t i = 0; i < weights.rows(); ++i)
                for(size_t j = 0; j < weights.cols(); ++j)
                    weights[i][j] = genrand();
            for(size_t i = 0; i < biases.rows(); ++i)
                for(size_t j = 0; j < biases.cols(); ++j)
                    biases[i][j] = 0.001;
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

        Tensor2D<T> eval(const Tensor2D<T>& input) const {
            return softmax(layer3.eval(layer2.eval(layer1.eval(input))));
        }

        void backward(Tensor2D<T> sm, const Tensor2D<unsigned int>& actual, const Tensor2D<T>& input) {
            
            // Backprop through softmax
            for(size_t r = 0; r < sm.rows(); ++r)
                sm[r][actual[r][0]] -= 1;

            for(size_t r = 0; r < sm.rows(); ++r)
                for(size_t c = 0; c < sm.cols(); ++c)
                    sm[r][c] /= sm.rows();

            // Backprop through layer3 
            layer3.weights_grad = dot(transpose(layer2.getacts()), sm);
            layer3.weights_grad = add(layer3.weights_grad, mul(layer3.weights, wt_reg));
            for(size_t r = 0; r < sm.rows(); ++r)
                for(size_t c = 0; c < sm.cols(); ++c)
                    layer3.biases_grad[0][c] += sm[r][c];

            // Backprop through layer2 
            auto hidden2 = dot(sm, transpose(layer3.weights));
            for(size_t r = 0; r < hidden2.rows(); ++r)
                for(size_t c = 0; c < hidden2.rows(); ++c)
                    if (layer2.getacts()[r][c] == 0)
                        hidden2[r][c] = 0.0;

            layer2.weights_grad = dot(transpose(layer1.getacts()), hidden2);
            layer2.weights_grad = add(layer2.weights_grad, mul(layer2.weights, wt_reg));
            for(size_t r = 0; r < hidden2.rows(); ++r)
                for(size_t c = 0; c < hidden2.cols(); ++c)
                    layer2.biases_grad[0][c] += hidden2[r][c];

            // Backprop through layer1 
            auto hidden1 = dot(hidden2, transpose(layer2.weights));
            for(size_t r = 0; r < hidden1.rows(); ++r)
                for(size_t c = 0; c < hidden1.rows(); ++c)
                    if (layer1.getacts()[r][c] == 0)
                        hidden1[r][c] = 0.0;

            layer1.weights_grad = dot(transpose(input), hidden1);
            layer1.weights_grad = add(layer1.weights_grad, mul(layer1.weights, wt_reg));
            for(size_t r = 0; r < hidden1.rows(); ++r)
                for(size_t c = 0; c < hidden1.cols(); ++c)
                    layer1.biases_grad[0][c] += hidden1[r][c];
        }

        void opt(float lr=learn_rate) {
            layer1.weights = sub(layer1.weights, mul(layer1.weights_grad, lr));
            layer2.weights = sub(layer2.weights, mul(layer2.weights_grad, lr));
            layer3.weights = sub(layer3.weights, mul(layer3.weights_grad, lr));

            layer1.biases = sub(layer1.biases, mul(layer1.biases_grad, lr));
            layer2.biases = sub(layer2.biases, mul(layer2.biases_grad, lr));
            layer3.biases = sub(layer3.biases, mul(layer3.biases_grad, lr));

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

float get_accuracy(const Network<precision>& nt, MNISTDataLoader& loader)
{
    batchtype data = loader.fetch(5000);
    auto sm = nt.eval(data.first);
    size_t totcorrect = 0;
    for(size_t r = 0; r < sm.rows(); ++r) {
        if(data.second[r][0] == maxidx(sm[r], sm.cols()))
            totcorrect++;
    }

    return (float)totcorrect/sm.rows();
}

void mnist()
{
    cout << "Starting MNIST training ..." << endl;
    MNISTDataLoader train(train_data, train_label);
    MNISTDataLoader test(test_data, test_label);
    Network<precision> nt(784, 10, 512, 1024);

    size_t epochs = num_epochs;
    size_t batches = train.numitems() / batch_size;
    size_t i = 1;

    while(i <= epochs) {
        size_t j = 1;
        while(j <= batches) {
            batchtype batch = train.fetch(batch_size);
            auto sm = nt.forward(batch.first);

            if (j%10 == 0) {
                float loss = logloss(batch.second, sm);
                float trainacc = get_accuracy(nt, train);
                float testacc = get_accuracy(nt, test);

                cout << "Ep:" << i << "/" <<epochs << ", Batch:"; 
                cout << j << "/" << batches << ") ";
                cout << setprecision(3) << fixed;
                cout << "Loss: " << loss << ", Train Acc: " << trainacc;
                cout << ", Test Acc: " << testacc << endl;
            }
            nt.backward(sm, batch.second, batch.first); 
            nt.opt(0.001);
            j++;
        }
        i++;
    }

}

