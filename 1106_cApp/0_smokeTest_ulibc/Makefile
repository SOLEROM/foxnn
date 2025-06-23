# ---------------------------------------------------------------------------
#  BUILD CONFIGURATION
# ---------------------------------------------------------------------------
CROSS_COMPILE=/home/user/shared/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-

CC      := $(CROSS_COMPILE)gcc
STRIP   := $(CROSS_COMPILE)strip

## for ver1.6
RKNN_SDK := /home/user/shared/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api


# ---------------------------------------------------------------------------
#  RKNN runtime – your exact SDK layout
# ---------------------------------------------------------------------------
RKNN_INC := $(RKNN_SDK)/include

## for armhf-uclibc
RKNN_LIB := $(RKNN_SDK)/armhf-uclibc
LDFLAGS  := -L$(RKNN_LIB) -lrknnmrt -lpthread -lm -ldl

# ---------------------------------------------------------------------------
#  Build flags
# ---------------------------------------------------------------------------
CFLAGS  := -O2 -Wall -I$(RKNN_INC)

TARGET  := npu_probe
SRC     := npu_probe.c
OBJ     := $(SRC:.c=.o)

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

install: $(TARGET)
	mkdir -p out
	$(STRIP) -s $< -o out/$<
	@echo "→ out/$(TARGET) ready."

clean:
	rm -f $(OBJ) $(TARGET)
	rm -rf out
