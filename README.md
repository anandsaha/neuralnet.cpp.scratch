# LeapMind

Neural net implementation in C++ from scratch. Trained MNIST with 3 layer network. Added weight regularization.

Could have implemented Dropout and Batch Norm, but would have taken more days.

Developed on Ubuntu 16.04 using g++ 5.4 `g++ (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609`

### Important Files

* `mnist.h`: Main file with all the code
* `main.cpp`: Calls the `mnist()` function in `mnist.h`, that's all
* `data`: Directory where MNIST files are kept
* `data\dl.sh`: Bash script to download and unzip the MNIST files
* `output.txt`: Execution log of a full run with 5 episodes 

### Download data

```
cd data
./dl.sh
cd ..
```

### Build and Run

Edit configurations in mnist.h (epochs, batch size etc.). Then,

```
make
./mnist
```

### Valgrind Report

```
==23034== 
==23034== HEAP SUMMARY:
==23034==     in use at exit: 72,704 bytes in 1 blocks
==23034==   total heap usage: 386,689 allocs, 386,688 frees, 673,798,516 bytes allocated
==23034== 
==23034== 72,704 bytes in 1 blocks are still reachable in loss record 1 of 1
==23034==    at 0x4C2DB8F: malloc (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==23034==    by 0x4EC6365: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.24)
==23034==    by 0x40106B9: call_init.part.0 (dl-init.c:72)
==23034==    by 0x40107CA: call_init (dl-init.c:30)
==23034==    by 0x40107CA: _dl_init (dl-init.c:120)
==23034==    by 0x4000C69: ??? (in /lib/x86_64-linux-gnu/ld-2.23.so)
==23034== 
==23034== LEAK SUMMARY:
==23034==    definitely lost: 0 bytes in 0 blocks
==23034==    indirectly lost: 0 bytes in 0 blocks
==23034==      possibly lost: 0 bytes in 0 blocks
==23034==    still reachable: 72,704 bytes in 1 blocks
==23034==         suppressed: 0 bytes in 0 blocks
==23034== 
==23034== For counts of detected and suppressed errors, rerun with: -v
==23034== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)

```
