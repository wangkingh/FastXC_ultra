# ----------------------------------------------------
#  RotateNCF ‒ 子目录 Makefile（Release / Debug 自动切换）
# ----------------------------------------------------

# ---------- 编译器 ----------
CC        ?= gcc
WARN      ?= -Wall

# -------- 优化级别自动切换 --------
ifeq ($(MAKELEVEL),0)           # 手动调用 → Debug
    CFLAGS ?= -O0 -g $(WARN)
else                            # 顶层批量构建 → Release
    CFLAGS ?= -O3 $(WARN)
endif
# ---------------------------------

# ---------- 链接库 ----------
LDFLAGS := -lm

# ---------- 源文件 ----------
OBJS = \
    main.o \
    arguproc.o \
    sacio.o \
    hddistance.o \
    rotate.o

# ---------- 目标可执行文件 ----------
BINDIR := ../../bin
PROG   := $(BINDIR)/RotateNCF

# ---------- 伪目标 ----------
.PHONY: all clean veryclean

# ---------- 默认目标 ----------
all: $(BINDIR) $(PROG)

# ---------- 创建输出目录 ----------
$(BINDIR):
	@mkdir -p $@

# ---------- 链接 ----------
$(PROG): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ---------- C 编译规则 ----------
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ---------- 头文件依赖 ----------
main.o: main.c sac.h
arguproc.o: arguproc.c sac.h
sacio.o: sacio.c sac.h
hddistance.o: hddistance.c sac.h
rotate.o: rotate.c sac.h

# ---------- 清理 ----------
clean:
	@rm -f *.o

veryclean: clean
	@rm -f $(PROG)
