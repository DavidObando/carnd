TARGET = h-as
MAIN_O = main.o
HYBRID_A_STAR_O = h-as.o
MAIN_CPP = main.cpp
HYBRID_A_STAR_H = hybrid_breadth_first.h
HYBRID_A_STAR_CPP = hybrid_breadth_first.cpp

$(TARGET): $(MAIN_O) $(HYBRID_A_STAR_O)
	g++ -g -o $(TARGET) $(MAIN_O) $(HYBRID_A_STAR_O) -std=c++11

$(MAIN_O): $(MAIN_CPP) $(HYBRID_A_STAR_H)
	g++ -g -o $(MAIN_O) -c $(MAIN_CPP) -std=c++11

$(HYBRID_A_STAR_O): $(HYBRID_A_STAR_CPP) $(HYBRID_A_STAR_H)
	g++ -g -o $(HYBRID_A_STAR_O) -c $(HYBRID_A_STAR_CPP) -std=c++11

all: $(TARGET)

clean:
	-rm $(TARGET) $(MAIN_O) $(HYBRID_A_STAR_O)
