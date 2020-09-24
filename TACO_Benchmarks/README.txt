This is a slightly updated version of the TACO paper.

1. Build Halide as usual (make, make distrib)
2. cd to the TACO_Benchmarks folder
3. either build each benchmark separately or all of them from the script (run_tests)
	--optionally change the arch_params to the ones specific for the platform
	  either from the script or by exporting HPARAMS (cap LLC size at 8MB, same as the default), 		  set (export) HL_NUM_THREADS to number of cores


