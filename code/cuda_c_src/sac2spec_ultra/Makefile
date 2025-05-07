# Compiler & Architecture
CC   ?= gcc                     # 如果环境里有 CC 就用环境变量，否则默认 gcc
NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_80

# compiler flags
CFLAGS=-Wall -O0 -g
# NVCCFLAGS = -g -G -arch=$(ARCH)
NVCCFLAGS = -O3 -arch=$(ARCH)


# linker flags
LFLAG=-L/usr/local/cuda/lib64  -lcufft -lcudart -lm

CUDAOBJS=cuda.rdcrtr.o \
	cuda.util.o \
	cuda.misc.o \
	cuda.processing.o \
	cuda.taper.o \
	cuda.main.o \

OBJS=design_filter_response.o \
	arguproc.o \
	cal_nseg.o \
	par_rw_data.o \
	read_filelist.o \
	sacio.o \
	util.o \
	
BINDIR=../../bin
PROG=$(BINDIR)/sac2spec_ultra

all: $(BINDIR) $(PROG)

$(BINDIR):
	mkdir -p $(BINDIR)

$(PROG):$(OBJS) $(CUDAOBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LFLAG)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

%.o: %.cu 
	$(NVCC) $(NVCCFLAGS) -c $<

.PHONY: clean veryclean

clean:
	-rm *.o

veryclean:
	-rm $(PROG) *.o 
