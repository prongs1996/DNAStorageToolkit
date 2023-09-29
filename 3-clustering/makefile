CC=g++
ODIR=obj
SDIR=src
IDIR=includes/
#CFLAGS= -O3 -I$(IDIR) -lpthread
CFLAGS=-O3 -I$(IDIR) -lpthread

### SRC and OBJ for q-gram.
_SRC=configparser.cpp parallel_clustering.cpp _editdistance.cpp
_OBJ=configparser.o parallel_clustering.o _editdistance.o

### SRC and OBJ for w-grams.
#_SRC=configparser.cpp parallel_clustering_w.cpp _editdistance.cpp
#_OBJ=configparser.o parallel_clustering_w.o _editdistance.o

_DEP=configparser.h
#_OBJ = $(pathsubst %.cpp,%.o,$(_SRC))
SRC=$(patsubst %,$(SDIR)/%,$(_SRC))
OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ))
DEP=$(patsubst %,$(IDIR)/%,$(_DEP))



exe/clustering:$(OBJ) 
	$(CC) -o $@ $^ $(CFLAGS)

run: exe/clustering
	./exe/clustering

$(ODIR)/%.o: $(SDIR)/%.cpp 
	$(CC) -o $@ -c $< $(CFLAGS)

checker: checker.py
	python3 checker.py $(ARGS)

inp: scripts/noise_string_gen.py
	python3 scripts/noise_string_gen.py $(ARGS)

.PHONY: clean
clean:
	rm -f $(ODIR)/*.o exe/*
