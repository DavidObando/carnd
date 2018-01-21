TARGET = classifier
MAIN_O = main.o
CLASSIFIER_O = classifier.o
MAIN_CPP = main.cpp
CLASSIFIER_H = classifier.h
CLASSIFIER_CPP = classifier.cpp

$(TARGET): $(MAIN_O) $(CLASSIFIER_O)
	g++ -g -o $(TARGET) $(MAIN_O) $(CLASSIFIER_O) -std=c++11

$(MAIN_O): $(MAIN_CPP) $(CLASSIFIER_H)
	g++ -g -o $(MAIN_O) -c $(MAIN_CPP) -std=c++11

$(CLASSIFIER_O): $(CLASSIFIER_CPP) $(CLASSIFIER_H)
	g++ -g -o $(CLASSIFIER_O) -c $(CLASSIFIER_CPP) -std=c++11

all: $(TARGET)

clean:
	-rm $(TARGET) $(MAIN_O) $(CLASSIFIER_O)
