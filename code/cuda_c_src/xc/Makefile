# ----------------------------------------------------
#  xc_fast ‒ 子目录 Makefile（Release / Debug 自动切换）
# ----------------------------------------------------

# ---------- 编译器 ----------
CC        ?= gcc
NVCC      ?= /usr/local/cuda/bin/nvcc
ARCH      ?= sm_80
WARN      ?= -Wall

# -------- 优化级别自动切换 --------
ifeq ($(MAKELEVEL),0)           # 手动单目录编译 → Debug
    CFLAGS     ?= -O0 -g $(WARN)
    NVCCFLAGS  ?= -g -G -lineinfo -arch=$(ARCH)
else                            # 顶层批量构建 → Release
    CFLAGS     ?= -O3 $(WARN)
    NVCCFLAGS  ?= -O3 --generate-line-info -arch=$(ARCH)
endif
# ---------------------------------

# ---------- 链接库 ----------
LFLAG = -L/usr/local/cuda/lib64 -lcufft -lcudart -lm

# ---------- 源文件 ----------
CUDAOBJS = \
    cuda.main.o \
    cuda.xc_dual.o \
    cuda.util.o

OBJS = \
    arguproc.o \
    sacio.o \
    cal_dist.o \
    gen_ccfpath.o \
    par_read_spec.o \
    par_write_sac.o \
    par_write_step.o \
    par_filter_nodes.o \
    my_write_sac.o \
    read_segspec.o \
    read_spec_lst.o \
    util.o

# ---------- 目标可执行文件 ----------
BINDIR := ../../bin
PROG   := $(BINDIR)/xc_fast

# ---------- 伪目标 ----------
.PHONY: all clean veryclean

# ---------- 默认目标 ----------
all: $(BINDIR) $(PROG)

# ---------- 创建输出目录 ----------
$(BINDIR):
	@mkdir -p $@

# ---------- 链接 ----------
$(PROG): $(OBJS) $(CUDAOBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LFLAG)

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
