TARGET = qps
MAIN_O = main.o
MAIN_CPP = main.cpp

$(TARGET): $(MAIN_O)
	g++ -g -o $(TARGET) $(MAIN_O) -std=c++11

$(MAIN_O): $(MAIN_CPP)
	g++ -g -o $(MAIN_O) -c $(MAIN_CPP) -std=c++11

all: $(TARGET)

clean:
	-rm $(TARGET) $(MAIN_O)
