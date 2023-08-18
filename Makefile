PRJNAME = ann

INCPATH = ./include
SRCPATH = ./src
BINPATH = ./bin
OBJPATH = ./obj
LIBPATH = ./lib
EXT_INC_FLAGS= -I/usr/include/mkl -I${HOME}/include
EXT_LIB_FLAGS= -L/usr/lib/x86_64-linux-gnu -L${HOME}/lib

RLS_LIB = $(PRJNAME)
DBG_LIB = $(PRJNAME)_dbg


CC = gcc
LD = gcc
AR = ar

COM_CFLAGS = -std=c11 -Wall -Wextra -I$(INCPATH) $(EXT_INC_FLAGS) -DINDEX_T=INT32 -DFEILD_T=FLT32
OPT_CFLAGS = -flto -O3

RLS_CFLAGS = -DNDEBUG $(COM_CFLAGS) $(OPT_CFLAGS)
RLS_LDFLAGS = $(OPT_CFLAGS) -L$(LIBPATH) $(EXT_LIB_FLAGS)
DBG_CFLAGS = -DDEBUG -g $(COM_CFLAGS) 
DBG_LDFLAGS = -L$(LIBPATH) $(EXT_LIB_FLAGS) -g
LD_LIBS = -llin_alg_flt32_dbg -lmkl_rt -lm #-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

CFILES = $(wildcard $(SRCPATH)/*.c)
HFILES = $(wildcard $(INCPATH)/*.h)
RLS_OBJS = $(patsubst $(SRCPATH)/%.c, $(OBJPATH)/%.o, $(CFILES))
DBG_OBJS = $(patsubst $(SRCPATH)/%.c, $(OBJPATH)/%_dbg.o, $(CFILES))

.PHONY: all clean release debug test run_test

all: debug release test
	@echo "====== make all ======"

$(OBJPATH)/%.o: $(SRCPATH)/%.c $(HFILES)
	@mkdir -p $(OBJPATH)
	$(CC) $(RLS_CFLAGS) -o $@ -c $<

$(OBJPATH)/%_dbg.o: $(SRCPATH)/%.c $(HFILES)
	@mkdir -p $(OBJPATH)
	$(CC) $(DBG_CFLAGS) -o $@ -c $<

$(LIBPATH)/lib$(RLS_LIB).a: $(RLS_OBJS)
	@mkdir -p $(LIBPATH)
	$(AR) rcs $@ $^

$(LIBPATH)/lib$(DBG_LIB).a: $(DBG_OBJS)
	@mkdir -p $(LIBPATH)
	$(AR) rcs $@ $^

$(BINPATH)/$(DBG_LIB)_test.out: $(SRCPATH)/$(PRJNAME)_test.c $(LIBPATH)/lib$(DBG_LIB).a
	@mkdir -p $(BINPATH)
	$(LD) $(DBG_LDFLAGS) -o $@ -l$(DBG_LIB) $(LD_LIBS)

release: $(LIBPATH)/lib$(RLS_LIB).a
	@echo "====== make release ======"

debug: $(LIBPATH)/lib$(DBG_LIB).a
	@echo "====== make debug ======"

test: $(BINPATH)/$(DBG_LIB)_test.out
	@echo "====== make test ======"

run_test: test
	$(BINPATH)/$(DBG_LIB)_test.out
	@echo "****** test finished ******"
	@echo "====== make run_test ======"

clean:
	rm -rf $(OBJPATH) $(BINPATH) $(LIBPATH)
	@echo "====== make clean ======"