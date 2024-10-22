pv197-proj.tar.gz: kernel_CPU.C kernel.cu framework.cu Makefile
	tar -czvf pv197-proj.tar.gz kernel_CPU.C kernel.cu framework.cu Makefile

bar: pv197-proj.tar.gz
	scp pv197-proj.tar.gz bar:proj/; echo "(cd ~/proj && tar -xzf pv197-proj.tar.gz && make run); exit" | ssh -tt bar;  

air: pv197-proj.tar.gz
	scp pv197-proj.tar.gz air:proj/; echo "(cd ~/proj && tar -xzf pv197-proj.tar.gz && make run); exit" | ssh -tt air;  




framework: kernel_CPU.C kernel.cu framework.cu
	nvcc -o framework framework.cu

run: framework
	./framework
