all:
	g++-7 -o mnist -pg main.cpp
	g++-7 -o test -pg test.cpp
    
clean:
	rm mnist

