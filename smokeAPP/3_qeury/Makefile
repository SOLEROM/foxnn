# ---------------------------------------------------------------------------
#  BUILD CONFIGURATION (x86 → ARM32 cross-compile for uClibc)
# ---------------------------------------------------------------------------
CROSS_COMPILE := /home/user/shared/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-
CC      := $(CROSS_COMPILE)gcc
CXX   := $(CROSS_COMPILE)g++
STRIP   := $(CROSS_COMPILE)strip

## for ver1.6
# RKNN_SDK := /home/user/shared/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api
# RKNN_INC := $(RKNN_SDK)/include
# RKNN_LIB := $(RKNN_SDK)/armhf-uclibc
## for ver2.3.2
RKNN_SDK := /home/user/shared/rknn-toolkit2-v2.3.2-2025-04-09/rknpu2/runtime/Linux/librknn_api
RKNN_INC := $(RKNN_SDK)/include
RKNN_LIB := $(RKNN_SDK)/armhf-uclibc

# ---------------------------------------------------------------------------
#  RKNN runtime – for uClibc-based armhf
# ---------------------------------------------------------------------------
RKNN_SO  := $(RKNN_LIB)/librknnmrt.so

LDFLAGS  := -L$(RKNN_LIB) -lrknnmrt -lpthread -lm -ldl -Wl,-rpath,'$$ORIGIN/lib'
CXXFLAGS   := -O2 -Wall -std=c++17  -I$(RKNN_INC)

TARGET := rknn_query
SRC    := rknn_query.cc
OBJ    := $(SRC:.cc=.o)

OUTDIR := out
LIBDIR := $(OUTDIR)/lib

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) $(LDFLAGS) -o $@

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CXX) $(CXXFLAGS) -x c -c $< -o $@

install: $(TARGET)
	mkdir -p $(LIBDIR)
	$(STRIP) -s $(TARGET) -o $(OUTDIR)/$(TARGET)
	cp -v $(RKNN_SO) $(LIBDIR)/
	@echo "✅ Installed: $(OUTDIR)/$(TARGET) and $(LIBDIR)/librknnrt.so"

clean:
	rm -f $(OBJ) $(TARGET)
	rm -rf $(OUTDIR)
