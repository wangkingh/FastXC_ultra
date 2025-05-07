export CC = gcc
export NVCC = /usr/local/cuda/bin/nvcc
export ARCH = sm_80

SUBDIRS = sac2spec_stable sac2spec_ultra xc rotate ncf_pws

.PHONY: $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	@echo "Making in directory $@"
	$(MAKE) -C $@

clean:
	for dir in $(SUBDIRS); do \
		echo "Cleaning in directory $$dir"; \
		$(MAKE) -C $$dir clean; \
	done

veryclean:
	for dir in $(SUBDIRS); do \
		echo "Performing very clean in directory $$dir"; \
		$(MAKE) -C $$dir veryclean; \
	done
