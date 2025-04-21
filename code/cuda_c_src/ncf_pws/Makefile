# Compiler & # cuda architecture
# CC=gcc
# NVCC=/usr/local/cuda/bin/nvcc
# ARCH = sm_89

# Compiler flags
CFLAGS=-I/usr/local/cuda/include
NVCCFLAGS=-g -G -arch=$(ARCH)

# Linker flags
LDFLAGS=-L/usr/local/cuda/lib64 -lcufft -lcudart -lm -L/usr/lib/x86_64-linux-gnu

CUDAOBJS=cuda.stransform.o \
		cuda.estimate_batch.o \
		cuda.pws_util.o \
		cuda.util.o \
		cuda.main.o

OBJS=arguproc.o \
	sacio.o \
	gen_ncf_path.o \
	util.o \
	read_big_sac.o

BINDIR=../../bin
PROG=$(BINDIR)/ncf_pws

all: $(BINDIR) $(PROG)

$(BINDIR):
	mkdir -p $(BINDIR)

$(PROG): $(OBJS) $(CUDAOBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $<

.PHONY: clean veryclean

clean:
	-rm *.o

veryclean:
	-rm $(PROG) *.o 

