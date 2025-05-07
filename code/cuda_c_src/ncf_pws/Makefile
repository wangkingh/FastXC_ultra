# ----------------------------------------------------
#  ncf_pws ‒ 子目录 Makefile（Release / Debug 自动切换）
# ----------------------------------------------------

# ---------- 编译器 ----------
CC        ?= gcc
NVCC      ?= /usr/local/cuda/bin/nvcc
ARCH      ?= sm_80
WARN      ?= -Wall

# ---------- 通用宏 ----------
BITS64    := -D_FILE_OFFSET_BITS=64
CINCLUDE  := -I/usr/local/cuda/include

# -------- 优化级别自动切换 --------
ifeq ($(MAKELEVEL),0)           # 手动单目录编译 → Debug
    CFLAGS     ?= -O0 -g $(WARN) $(BITS64) $(CINCLUDE)
    NVCCFLAGS  ?= -g -G -lineinfo -arch=$(ARCH) -Xcompiler "$(BITS64)" $(CINCLUDE)
else                            # 顶层批量构建 → Release
    CFLAGS     ?= -O3 $(WARN) $(BITS64) $(CINCLUDE)
    NVCCFLAGS  ?= -O3 --generate-line-info -arch=$(ARCH) -Xcompiler "$(BITS64)" $(CINCLUDE)
endif
# ---------------------------------

# ---------- 链接库 ----------
LDFLAGS = -L/usr/local/cuda/lib64 -lcufft -lcudart -lm -L/usr/lib/x86_64-linux-gnu

# ---------- 源文件 ----------
CUDAOBJS = \
    cuda.stransform.o \
    cuda.estimate_batch.o \
    cuda.pws_util.o \
    cuda.util.o \
    cuda.main.o

OBJS = \
    arguproc.o \
    sacio.o \
    prestack.o \
    gen_ncf_path.o \
    util.o \
    big_sacio.o

# ---------- 目标可执行文件 ----------
BINDIR := ../../bin
PROG   := $(BINDIR)/ncf_pws

# ---------- 伪目标 ----------
.PHONY: all clean veryclean

# ---------- 默认目标 ----------
all: $(BINDIR) $(PROG)

# ---------- 创建输出目录 ----------
$(BINDIR):
	@mkdir -p $@

# ---------- 链接 ----------
$(PROG): $(OBJS) $(CUDAOBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# ---------- C 源编译 ----------
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ---------- CUDA 源编译 ----------
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ---------- 清理 ----------
clean:
	@rm -f *.o

veryclean: clean
	@rm -f $(PROG)
