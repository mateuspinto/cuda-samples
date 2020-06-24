CC =nvcc
CFLAGS =-arch=compute_61 -O2
BINFOLDER =bin

SOURCES = $(wildcard *.cu)
EXECS = $(SOURCES:%.cu=%)

all: $(EXECS)

FOLDER:
	mkdir -p $(BINFOLDER)

$(EXECS): %: %.cu
	$(CC) $(CFLAGS) -o $(BINFOLDER)/$@ $<
