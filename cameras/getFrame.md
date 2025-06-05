# save frame

```
v4l2-ctl --list-devices
[root@luckfox ]# v4l2-ctl --list-devices
rkisp-statistics (platform: rkisp):
	/dev/video19
	/dev/video20

rkcif-mipi-lvds (platform:rkcif):
	/dev/media0

rkcif (platform:rkcif-mipi-lvds):
	/dev/video0
	/dev/video1
	/dev/video2
	/dev/video3
	/dev/video4
	/dev/video5
	/dev/video6
	/dev/video7
	/dev/video8
	/dev/video9
	/dev/video10

rkisp_mainpath (platform:rkisp-vir0):
	/dev/video11
	/dev/video12
	/dev/video13
	/dev/video14
	/dev/video15
	/dev/video16
	/dev/video17
	/dev/video18
	/dev/media1
```

Pico 1160 is using the Rockchip ISP (rkisp) pipeline, which is a media-controller-based video device system.

(sensor → ISP → output node), and then capture from the appropriate rkisp_mainpath node

```
[root@luckfox ]# v4l2-ctl --device=/dev/video0 --list-formats-ext
ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture Multiplanar

	[0]: 'RG10' (10-bit Bayer RGRG/GBGB)
		Size: Stepwise 64x64 - 2304x1296 with step 8/8
	[1]: 'BA10' (10-bit Bayer GRGR/BGBG)
		Size: Stepwise 64x64 - 2304x1296 with step 8/8
	[2]: 'GB10' (10-bit Bayer GBGB/RGRG)
		Size: Stepwise 64x64 - 2304x1296 with step 8/8
	[3]: 'BG10' (10-bit Bayer BGBG/GRGR)
		Size: Stepwise 64x64 - 2304x1296 with step 8/8
	[4]: 'Y10 ' (10-bit Greyscale)
		Size: Stepwise 64x64 - 2304x1296 with step 8/8
```



/oem/usr/bin/RkLunch-stop.sh


WIDTH=1280       
HEIGHT=720
v4l2-ctl -d /dev/video11 \
         --set-fmt-video=width=$WIDTH,height=$HEIGHT,pixelformat=NV12 \
         --stream-mmap --stream-count=1 --stream-to=frame.nv12