# ---------------------------------------------------------------------------
#  BUILD CONFIGURATION (x86 → ARM32 cross-compile for uClibc)
# ---------------------------------------------------------------------------
CROSS_COMPILE := /home/user/shared/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-
CC      := $(CROSS_COMPILE)gcc
STRIP   := $(CROSS_COMPILE)strip

## for ver1.6
RKNN_SDK := /home/user/shared/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api
RKNN_INC := $(RKNN_SDK)/include
RKNN_LIB := $(RKNN_SDK)/armhf-uclibc
## for ver2.3
# RKNN_SDK := /home/user/shared/rknn_model_zoo/install/rv1106_linux_armv7l/rknn_yolov5_demo/
# RKNN_INC := /home/user/shared/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include
# RKNN_LIB := $(RKNN_SDK)/lib

# ---------------------------------------------------------------------------
#  RKNN runtime – for uClibc-based armhf
# ---------------------------------------------------------------------------
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
