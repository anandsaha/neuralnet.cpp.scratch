all:
	g++ -std=c++11 -o mnist -pg main.cpp
	g++ -std=c++11 -o test -pg test.cpp
    
clean:
	rm mnist test

