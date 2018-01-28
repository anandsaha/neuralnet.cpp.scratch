# leapmind

Neural net implementation in C++ from scratch. Trained MNIST with 3 layer network. Added weight regularization.

Developed on Ubuntu 16.04 using g++ 5.4 `g++ (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609`

### Important Files

`mnist.h`: Main file with all the code
`main.cpp`: Calls the `mnist()` function in `mnist.h`, that's all
`data`: Directory where MNIST files are kept
`data\dl.sh`: Bash script to download and unzip the MNIST files

### Download data

```
cd data
./dl.sh
cd ..
```

### Build and Run

```
make
./mnist
```

