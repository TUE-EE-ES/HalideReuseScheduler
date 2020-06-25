include ../../apps/support/Makefile.inc

BIN ?= bin

all: $(BIN)/process

$(BIN)/interpolate.generator: interpolate_generator.cpp $(GENERATOR_DEPS)
	@-mkdir -p $(BIN)
	@$(CXX) $(CXXFLAGS) -g -fno-rtti $(filter-out %.h,$^) -o $@ $(LDFLAGS)

$(BIN)/interpolate.a: $(BIN)/interpolate.generator
	@-mkdir -p $(BIN)
	@$^ -g interpolate -o $(BIN) -f interpolate target=$(HL_TARGET) auto_schedule=false

$(BIN)/interpolate_auto_schedule.a: $(BIN)/interpolate.generator
	@-mkdir -p $(BIN)
	@$^ -g interpolate -o $(BIN) -f interpolate_auto_schedule target=$(HL_TARGET)-no_runtime auto_schedule=true -e static_library,h,schedule machine_params=$(HPARAMS) > /dev/null 2>&1 

$(BIN)/process: process.cpp $(BIN)/interpolate.a $(BIN)/interpolate_auto_schedule.a
	@-mkdir -p $(BIN)
	@$(CXX) $(CXXFLAGS) -I$(BIN) -Wall -O3 $^ -o $@ $(LDFLAGS)

run: $(BIN)/process
	@-mkdir -p $(BIN)
	$(BIN)/process

clean:
	rm -rf $(BIN)

test: run
