watch:
	mkdir -p build && cd build && cmake .. -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DFASTBUILD:BOOL=ON && ls ../* | entr -s "rm -f ./run_bitlora && make -j40 run_bitlora && ./run_bitlora"

run:
	mkdir -p build && cd build && cmake .. -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DFASTBUILD:BOOL=ON && rm -f ./run_bitlora && make -j40 run_bitlora && ./run_bitlora
