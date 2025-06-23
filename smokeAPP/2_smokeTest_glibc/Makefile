# ---------------------------------------------------------------------------
#  BUILD CONFIGURATION (x86 → ARM32 cross-compile for glibc)
# ---------------------------------------------------------------------------
## from apt install
CROSS_COMPILE := arm-linux-gnueabihf-
CC      := $(CROSS_COMPILE)gcc
STRIP   := $(CROSS_COMPILE)strip

## for ver1.6
RKNN_SDK := /home/user/shared/luckfox-pico/project/cfg/BoardConfig_IPC/overlay/overlay-luckfox-ubuntu-rockchip/usr

# ---------------------------------------------------------------------------
#  RKNN runtime – for glibc-based armhf
# ---------------------------------------------------------------------------
## for rknn_api.h
RKNN_INC := /home/user/shared/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include

## for glibc
RKNN_LIB := $(RKNN_SDK)/lib
RKNN_SO  := $(RKNN_LIB)/librknnmrt.so         

LDFLAGS  := -L$(RKNN_LIB) -lrknnmrt -lpthread -lm -ldl -Wl,-rpath,'$$ORIGIN/lib'
CFLAGS   := -O2 -Wall -I$(RKNN_INC)

TARGET := npu_probe
SRC    := npu_probe.c
OBJ    := $(SRC:.c=.o)

OUTDIR := out
LIBDIR := $(OUTDIR)/lib

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

install: $(TARGET)
	mkdir -p $(LIBDIR)
	$(STRIP) -s $(TARGET) -o $(OUTDIR)/$(TARGET)
	cp -v $(RKNN_SO) $(LIBDIR)/
	@echo "✅ Installed: $(OUTDIR)/$(TARGET) and $(LIBDIR)/librknnrt.so"

clean:
	rm -f $(OBJ) $(TARGET)
	rm -rf $(OUTDIR)
