include ../../apps/support/Makefile.inc

BIN ?= bin

all: $(BIN)/process

$(BIN)/unsharp.generator: unsharp_generator.cpp $(GENERATOR_DEPS)
	@-mkdir -p $(BIN)
	@$(CXX) $(CXXFLAGS) -g -fno-rtti $(filter-out %.h,$^) -o $@ $(LDFLAGS)

$(BIN)/unsharp.a: $(BIN)/unsharp.generator
	@-mkdir -p $(BIN)
	@$^ -g unsharp -o $(BIN) -f unsharp target=$(HL_TARGET) auto_schedule=false

$(BIN)/unsharp_auto_schedule.a: $(BIN)/unsharp.generator
	@-mkdir -p $(BIN)
	@$^ -g unsharp -o $(BIN) -f unsharp_auto_schedule target=$(HL_TARGET)-no_runtime auto_schedule=true -e static_library,h,schedule machine_params=$(HPARAMS) > /dev/null 2>&1 

$(BIN)/process: process.cpp $(BIN)/unsharp.a $(BIN)/unsharp_auto_schedule.a
	@-mkdir -p $(BIN)
	@$(CXX) $(CXXFLAGS) -I$(BIN) -Wall -O3 $^ -o $@ $(LDFLAGS)

run: $(BIN)/process
	@-mkdir -p $(BIN)
	$(BIN)/process

clean:
	@rm -rf $(BIN)

test: run
