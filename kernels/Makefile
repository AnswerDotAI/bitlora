NUM_JOBS ?= 40

watch:
	mkdir -p build && cd build && cmake .. -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DFASTBUILD:BOOL=ON && ls ../* | entr -s "rm -f ./run_bitlora && make -j$(NUM_JOBS) run_bitlora && ./run_bitlora"

run:
	mkdir -p build && cd build && cmake .. -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DFASTBUILD:BOOL=ON && rm -f ./run_bitlora && make -j$(NUM_JOBS) run_bitlora && ./run_bitlora

clean:
	read -r -p "Are you sure? [CTRL-C to abort] " response && rm -rf build/* 
