
all:
	g++ -O3 -Wall  -std=c++11 -o mnist -g main.cpp
	g++ -O3 -Wall  -std=c++11 -o test  -g test.cpp
    
clean:
	rm mnist test

