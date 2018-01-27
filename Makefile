all:
	g++ -O3 -std=c++11 -o mnist main.cpp
	g++ -O3 -std=c++11 -o test  test.cpp
    
clean:
	rm mnist test

