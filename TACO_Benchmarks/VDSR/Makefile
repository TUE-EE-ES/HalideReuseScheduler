include ../../apps/support/Makefile.inc

BIN ?= bin

all: $(BIN)/process

$(BIN)/vdsr_input.generator: VDSR_input_generator.cpp $(GENERATOR_DEPS)
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -g -fno-rtti $(filter-out %.h,$^) -o $@ $(LDFLAGS)

$(BIN)/vdsr_first_layer.generator: VDSR_first_layer_generator.cpp $(GENERATOR_DEPS)
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -g -fno-rtti $(filter-out %.h,$^) -o $@ $(LDFLAGS)

$(BIN)/vdsr_convnet.generator: VDSR_conv_net_generator.cpp $(GENERATOR_DEPS)
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -g -fno-rtti $(filter-out %.h,$^) -o $@ $(LDFLAGS)

$(BIN)/vdsr_final_layer.generator: VDSR_final_layer_generator.cpp $(GENERATOR_DEPS)
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -g -fno-rtti $(filter-out %.h,$^) -o $@ $(LDFLAGS)

$(BIN)/vdsr_output.generator: VDSR_output_generator.cpp $(GENERATOR_DEPS)
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -g -fno-rtti $(filter-out %.h,$^) -o $@ $(LDFLAGS)


$(BIN)/vdsr_input.a: $(BIN)/vdsr_input.generator
	@-mkdir -p $(BIN)
	$^ -g vdsr_input -o $(BIN) -f vdsr_input target=$(HL_TARGET) auto_schedule=false

$(BIN)/vdsr_first_layer.a: $(BIN)/vdsr_first_layer.generator
	@-mkdir -p $(BIN)
	$^ -g vdsr_first_layer -o $(BIN) -f vdsr_first_layer target=$(HL_TARGET)-no_runtime auto_schedule=false

$(BIN)/vdsr_convnet.a: $(BIN)/vdsr_convnet.generator
	@-mkdir -p $(BIN)
	$^ -g vdsr_convnet -o $(BIN) -f vdsr_convnet target=$(HL_TARGET)-no_runtime-profile auto_schedule=true machine_params=$(HPARAMS)

$(BIN)/vdsr_final_layer.a: $(BIN)/vdsr_final_layer.generator
	@-mkdir -p $(BIN)
	$^ -g vdsr_final_layer -o $(BIN) -f vdsr_final_layer target=$(HL_TARGET)-no_runtime auto_schedule=false

$(BIN)/vdsr_output.a: $(BIN)/vdsr_output.generator
	@-mkdir -p $(BIN)
	$^ -g vdsr_output -o $(BIN) -f vdsr_output target=$(HL_TARGET)-no_runtime auto_schedule=false



$(BIN)/process: VDSR_run.cpp $(BIN)/vdsr_input.a $(BIN)/vdsr_first_layer.a $(BIN)/vdsr_convnet.a $(BIN)/vdsr_final_layer.a $(BIN)/vdsr_output.a
	@-mkdir -p $(BIN)
	$(CXX) -g $(CXXFLAGS) -I$(BIN) -Wall -O3 $^ -o $@ $(LDFLAGS) -ljpeg -lpng

run: $(BIN)/process
	@-mkdir -p $(BIN)
	$(BIN)/process

clean:
	rm -rf $(BIN)

test: run
