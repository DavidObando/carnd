TARGET = cost
MAIN_O = main.o
COST_O = cost.o
MAIN_CPP = main.cpp
COST_H = cost.h
COST_CPP = cost.cpp

$(TARGET): $(MAIN_O) $(COST_O)
	g++ -g -o $(TARGET) $(MAIN_O) $(COST_O) -std=c++11

$(MAIN_O): $(MAIN_CPP) $(COST_H)
	g++ -g -o $(MAIN_O) -c $(MAIN_CPP) -std=c++11

$(COST_O): $(COST_CPP) $(COST_H)
	g++ -g -o $(COST_O) -c $(COST_CPP) -std=c++11

all: $(TARGET)

clean:
	-rm $(TARGET) $(MAIN_O) $(COST_O)
