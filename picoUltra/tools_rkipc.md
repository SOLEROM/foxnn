# rkipc Tool Usage


```
/oem/usr/bin/rkipc --help
[common.c][rkipc_version_dump]:rkipc version: unknown rkipc version for missing VCS info
[common.c][rkipc_version_dump]:rkipc info: unknown rkipc build info
[common.c][rkipc_version_dump]:rkipc type: COMPILE_FOR_RV1106_IPC
Usage: /oem/usr/bin/rkipc [options]
Version V1.0
Options:
-c | --config      rkipc ini file, default is /userdata/rkipc.ini, need to be writable
-a | --aiq_file    aiq file dir path, default is /etc/iqfiles
-l | --log_level   log_level [0/1/2/3], default is 2
-h | --help        for help 

```


## ??? more tools

```
[root@luckfox root]# rk_
rk_adc_test           rk_led_test           rk_mpi_af_test        rk_mpi_ao_test        rk_mpi_dup_venc_test  rk_mpi_rgn_test       rk_mpi_venc_test      rk_mpi_vpss_test      rk_system_test
rk_event_test         rk_mpi_adec_test      rk_mpi_ai_test        rk_mpi_avio_test      rk_mpi_mb_test        rk_mpi_sys_test       rk_mpi_vi_dup_test    rk_pwm_test           rk_time_test
rk_gpio_test          rk_mpi_aenc_test      rk_mpi_amix_test      rk_mpi_avs_test       rk_mpi_mmz_test       rk_mpi_tde_test       rk_mpi_vi_test        rk_rve_sample_test    rk_watchdog_test

```


## process???

```


  461 root     rkipc -a /oem/usr/share/iqfiles
  464 root     rkwifi_server start

```
