all:
	$(MAKE) -C ../.. KPATH=/usr/src/linux-headers-6.8.0-83-generic KBUILDTOP=$(abspath .) kernel
clean:
	$(MAKE) -C ../.. KPATH=/usr/src/linux-headers-6.8.0-83-generic KBUILDTOP=$(abspath .) clean_kernel
.PHONY: all clean
