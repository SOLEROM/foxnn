/********************************************************************
 * yolov5_rknn_init.cpp  –  full introspection + dummy inference
 *
 * Build:
 *   g++ -std=c++17 yolov5_rknn_init.cpp -lrknn_api -o yolov5_rknn_init
 *
 * Run:
 *   ./yolov5_rknn_init model.rknn
 ********************************************************************/
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <rknn_api.h>

/* ───────── helpers ───────── */
const char* fmt_str(int fmt) {
    switch (fmt) {
        case RKNN_TENSOR_NCHW: return "NCHW";
        case RKNN_TENSOR_NHWC: return "NHWC";
        default: return "UNKNOWN_FMT";
    }
}
const char* type_str(int type) {
    switch (type) {
        case RKNN_TENSOR_FLOAT32: return "float32";
        case RKNN_TENSOR_INT8:    return "int8";
        case RKNN_TENSOR_UINT8:   return "uint8";
        default: return "UNKNOWN_TYPE";
    }
}
const char* qnt_str(int q) {
    switch (q) {
        case RKNN_TENSOR_QNT_NONE:              return "NONE";
        case RKNN_TENSOR_QNT_DFP:               return "DFP";
        case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC: return "ASYM (u8/i8)";
        default: return "UNKNOWN_QNT_TYPE";
    }
}
void dump_attr(const rknn_tensor_attr& a) {
    std::cout << "    index="   << a.index
              << ", name="      << a.name
              << ", n_dims="    << a.n_dims
              << ", dims=["     << a.dims[0] << ',' << a.dims[1]
                               << ',' << a.dims[2] << ',' << a.dims[3] << ']'
              << ", elems="     << a.n_elems
              << ", size="      << a.size
              << ", fmt="       << fmt_str(a.fmt)
              << ", type="      << type_str(a.type)
              << ", qnt="       << qnt_str(a.qnt_type)
              << ", zp="        << a.zp
              << ", scale="     << a.scale
              << '\n';
}

/* ───────── main ───────── */
int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " model.rknn\n";
        return 1;
    }
    const char* model_path = argv[1];

    /* read model into memory */
    std::ifstream ifs(model_path, std::ios::binary | std::ios::ate);
    if (!ifs) { std::cerr << "Cannot open " << model_path << '\n'; return 1; }
    size_t model_size = ifs.tellg();
    std::vector<uint8_t> model_buf(model_size);
    ifs.seekg(0);  ifs.read(reinterpret_cast<char*>(model_buf.data()), model_size);

    /* init RKNN */
    rknn_context ctx;
    if (rknn_init(&ctx, model_buf.data(), model_size, 0, nullptr) != RKNN_SUCC) {
        std::cerr << "rknn_init failed\n"; return 1;
    }

    /* SDK / driver versions */
    rknn_sdk_version ver;
    if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == RKNN_SUCC)
        std::cout << "SDK: " << ver.api_version << "  |  Driver: " << ver.drv_version << "\n";

    /* I/O counts */
    rknn_input_output_num io_num;
    if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) != RKNN_SUCC) {
        std::cerr << "RKNN_QUERY_IN_OUT_NUM failed\n"; return 1;
    }
    std::cout << "\nI/O counts → inputs=" << io_num.n_input
              << "  outputs=" << io_num.n_output << '\n';

    /* ── input attributes ── */
    std::vector<rknn_tensor_attr> in_attr(io_num.n_input);
    std::cout << "\nInput tensors:\n";
    for (int i = 0; i < io_num.n_input; ++i) {
        std::memset(&in_attr[i], 0, sizeof(rknn_tensor_attr));
        in_attr[i].index = i;
        if (rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &in_attr[i], sizeof(rknn_tensor_attr)) != RKNN_SUCC) {
            std::cerr << "RKNN_QUERY_NATIVE_INPUT_ATTR failed on idx " << i << '\n'; return 1;
        }
        dump_attr(in_attr[i]);
    }

    /* ── output attributes ── */
    std::vector<rknn_tensor_attr> out_attr(io_num.n_output);
    std::cout << "\nOutput tensors:\n";
    for (int i = 0; i < io_num.n_output; ++i) {
        std::memset(&out_attr[i], 0, sizeof(rknn_tensor_attr));
        out_attr[i].index = i;
        if (rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr[i], sizeof(rknn_tensor_attr)) != RKNN_SUCC) {
            std::cerr << "RKNN_QUERY_OUTPUT_ATTR failed on idx " << i << '\n'; return 1;
        }
        dump_attr(out_attr[i]);
    }

    /* ── zero-copy buffers ── */
    // Force first input to UINT8 / NHWC so that norm-quant fuse happens in NPU
    in_attr[0].type = RKNN_TENSOR_UINT8;
    in_attr[0].fmt  = RKNN_TENSOR_NHWC;

    std::vector<rknn_tensor_mem*> in_mems(io_num.n_input, nullptr);
    std::vector<rknn_tensor_mem*> out_mems(io_num.n_output, nullptr);

    in_mems[0] = rknn_create_mem(ctx, in_attr[0].size_with_stride);
    if (rknn_set_io_mem(ctx, in_mems[0], &in_attr[0]) != RKNN_SUCC) {
        std::cerr << "rknn_set_io_mem (input 0) failed\n"; return 1;
    }
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        out_mems[i] = rknn_create_mem(ctx, out_attr[i].size_with_stride);
        if (rknn_set_io_mem(ctx, out_mems[i], &out_attr[i]) != RKNN_SUCC) {
            std::cerr << "rknn_set_io_mem (output " << i << ") failed\n"; return 1;
        }
    }

    /* ── report model geometry ── */
    int model_h, model_w, model_c;
    if (in_attr[0].fmt == RKNN_TENSOR_NCHW) {
        model_c = in_attr[0].dims[1];
        model_h = in_attr[0].dims[2];
        model_w = in_attr[0].dims[3];
        std::cout << "\nModel input format: NCHW\n";
    } else {
        model_h = in_attr[0].dims[1];
        model_w = in_attr[0].dims[2];
        model_c = in_attr[0].dims[3];
        std::cout << "\nModel input format: NHWC\n";
    }
    std::cout << "Input shape: H=" << model_h << "  W=" << model_w << "  C=" << model_c << '\n';

    /* ────────────────────────────────────
     *  DUMMY RUN – feed constant data and
     *  print raw outputs (int8 / uint8)
     * ────────────────────────────────── */
    {
        std::cout << "\nRunning dummy inference …\n";

        /* fill input[0] with mid-level 127 */
        auto* in_data = static_cast<uint8_t*>(in_mems[0]->virt_addr);
        std::fill_n(in_data, in_attr[0].size_with_stride, 127);

        /* run once */
        if (rknn_run(ctx, nullptr) != RKNN_SUCC) {
            std::cerr << "rknn_run failed\n"; goto CLEANUP;
        }

        /* dump each output */
        for (uint32_t i = 0; i < io_num.n_output; ++i) {
            std::cout << "\n── Output #" << i << " ──\n";
            const int8_t* odata = static_cast<const int8_t*>(out_mems[i]->virt_addr);
            size_t nelems       = out_attr[i].n_elems;

            /* print first 8 elements */
            std::cout << "first 8 values : ";
            for (size_t j = 0; j < std::min<size_t>(8, nelems); ++j)
                std::cout << int(odata[j]) << ' ';
            std::cout << '\n';

            /* basic stats */
            auto [imin, imax] = std::minmax_element(odata, odata + nelems);
            long long sum = std::accumulate(odata, odata + nelems, 0LL);
            std::cout << "min=" << int(*imin)
                      << "  max=" << int(*imax)
                      << "  sum=" << sum << '\n';
        }
    }

CLEANUP:
    /* clean-up */
    for (auto* m : in_mems)  if (m) rknn_destroy_mem(ctx, m);
    for (auto* m : out_mems) if (m) rknn_destroy_mem(ctx, m);
    rknn_destroy(ctx);
    return 0;
}
