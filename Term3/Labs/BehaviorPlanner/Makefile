TARGET = bp
MAIN_O = bp.o
ROAD_O = road.o
VEHICLE_O = vehicle.o
MAIN_CPP = main.cpp
ROAD_H = road.h
ROAD_CPP = road.cpp
VEHICLE_H = vehicle.h
VEHICLE_CPP = vehicle.cpp

$(TARGET): $(MAIN_O) $(ROAD_O) $(VEHICLE_O)
	g++ -g -o $(TARGET) $(MAIN_O) $(ROAD_O) $(VEHICLE_O) -std=c++11

$(MAIN_O): $(MAIN_CPP)
	g++ -g -o $(MAIN_O) -c $(MAIN_CPP) -std=c++11

$(ROAD_O): $(ROAD_CPP) $(ROAD_H)
	g++ -g -o $(ROAD_O) -c $(ROAD_CPP) -std=c++11

$(VEHICLE_O): $(VEHICLE_CPP) $(VEHICLE_H)
	g++ -g -o $(VEHICLE_O) -c $(VEHICLE_CPP) -std=c++11

all: $(TARGET)

clean:
	-rm $(TARGET) $(MAIN_O) $(ROAD_O) $(VEHICLE_O)
