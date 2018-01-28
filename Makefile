all:
	g++-7 -Wall -O3 -std=c++11 -o mnist -g main.cpp
	g++-7 -Wall -O3 -std=c++11 -o test  -g test.cpp
    
clean:
	rm mnist test

