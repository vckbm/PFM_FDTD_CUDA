cpu:
	g++ -o cpu cpu_FDTD.c -fopenmp -lpthread
gpu:
	nvcc -o gpu gpu_FDTD.cu

clean:
	rm *.txt *.png

restart:
	rm *.txt
	rm gpu
	make gpu
	./gpu
