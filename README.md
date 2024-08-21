# helix_cuda

# kompilace
nvcc main.cu -arch=sm_86

používám alias comp="nvcc main.cu -O3 -arch=sm_86; cp a.out input_files; cd input_files; ./a.out; cd .."
