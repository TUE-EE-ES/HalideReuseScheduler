include ../../apps/support/Makefile.inc

BIN ?= bin

all: $(BIN)/process

$(BIN)/maxfilter.generator: max_filter_generator.cpp $(GENERATOR_DEPS)
	@-mkdir -p $(BIN)
	@$(CXX) $(CXXFLAGS) -g -fno-rtti $(filter-out %.h,$^) -o $@ $(LDFLAGS)

$(BIN)/maxfilter.a: $(BIN)/maxfilter.generator
	@-mkdir -p $(BIN)
	@$^ -g maxfilter -o $(BIN) -f maxfilter target=$(HL_TARGET) auto_schedule=false

$(BIN)/maxfilter_auto_schedule.a: $(BIN)/maxfilter.generator
	@-mkdir -p $(BIN)
	@$^ -g maxfilter -o $(BIN) -f maxfilter_auto_schedule target=$(HL_TARGET)-no_runtime auto_schedule=true -e static_library,h,schedule machine_params=$(HPARAMS) 

$(BIN)/process: process.cpp $(BIN)/maxfilter.a $(BIN)/maxfilter_auto_schedule.a
	@-mkdir -p $(BIN)
	@$(CXX) $(CXXFLAGS) -I$(BIN) -Wall -O3 $^ -o $@ $(LDFLAGS)

run: $(BIN)/process
	@-mkdir -p $(BIN)
	$(BIN)/process

clean:
	@rm -rf $(BIN)

test: run
