include ../../apps/support/Makefile.inc

BIN ?= bin

all: $(BIN)/process

$(BIN)/harris.generator: harris_generator.cpp $(GENERATOR_DEPS)
	@-mkdir -p $(BIN)
	@$(CXX) $(CXXFLAGS) -ldl -lz -g -fno-rtti $(filter-out %.h,$^) -o $@ $(LDFLAGS)

$(BIN)/harris.a: $(BIN)/harris.generator
	@-mkdir -p $(BIN)
	@$^ -g harris -o $(BIN) -f harris target=$(HL_TARGET) auto_schedule=false

$(BIN)/harris_auto_schedule.a: $(BIN)/harris.generator
	@-mkdir -p $(BIN)
	@$^ -g harris -o $(BIN) -f harris_auto_schedule target=$(HL_TARGET)-no_runtime auto_schedule=true -e static_library,h,schedule machine_params=$(HPARAMS) 

$(BIN)/process: process.cpp $(BIN)/harris.a $(BIN)/harris_auto_schedule.a
	@-mkdir -p $(BIN)
	@$(CXX) $(CXXFLAGS) -I$(BIN) -Wall -O3 $^ -o $@ $(LDFLAGS)

run: $(BIN)/process
	@-mkdir -p $(BIN)
	$(BIN)/process

clean:
	rm -rf $(BIN)

test: run
