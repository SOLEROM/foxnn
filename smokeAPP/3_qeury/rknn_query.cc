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
#include <iomanip>     
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


/********************************************************************
 * Dump a UINT8 / INT8 input tensor in either NHWC or NCHW layout.
 * - attr must be the SAME rknn_tensor_attr you used for that tensor.
 ********************************************************************/
void print_input_tensor(const uint8_t* data, const rknn_tensor_attr& attr,
                        int max_lines_per_ch = 8)
{
    const int batch = attr.dims[0];
    if (batch != 1) {
        std::cout << "(print_input_tensor) Only batch-1 is handled\n";
        return;
    }

    int H, W, C;
    bool nchw = (attr.fmt == RKNN_TENSOR_NCHW);
    if (nchw) {            // NCHW
        C = attr.dims[1];  H = attr.dims[2];  W = attr.dims[3];
    } else {               // NHWC
        H = attr.dims[1];  W = attr.dims[2];  C = attr.dims[3];
    }

    std::cout << "\n── Dump of input tensor (" << (nchw ? "NCHW" : "NHWC")
              << ")  shape=[1," << H << ',' << W << ',' << C << "] ──\n";

    for (int c = 0; c < C; ++c) {
        std::cout << "[channel " << c << "]\n";
        int lines = 0;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                size_t idx;
                if (nchw)
                    idx = ((size_t)c * H + y) * W + x;          // N C H W → flat
                else
                    idx = ((size_t)y * W + x) * C + c;          // N H W C → flat

                std::cout << std::setw(3) << int(data[idx]) << ' ';
            }
            std::cout << '\n';
            if (++lines >= max_lines_per_ch && H > max_lines_per_ch) {
                std::cout << "  … (" << (H - max_lines_per_ch)
                          << " more rows)\n";
                break;
            }
        }
    }
    std::cout << "──────────────────────────────────────────────\n";
}


/********************************************************************
 * print_tensor_raw ― dump every element of an RKNN tensor
 *   • Handles UINT8, INT8, and FLOAT32
 *   • Prints values in flat (row-major) order, one line 16 elements wide
 ********************************************************************/
void print_tensor_raw(const rknn_tensor_mem* mem,
                      const rknn_tensor_attr& attr)
{
    const size_t n = attr.n_elems;               // total elements

    auto newline_every = [](size_t i) { return (i & 15) == 15; };

    switch (attr.type)
    {
    case RKNN_TENSOR_UINT8:
    {
        const uint8_t* d = static_cast<const uint8_t*>(mem->virt_addr);
        for (size_t i = 0; i < n; ++i) {
            std::cout << std::setw(4) << int(d[i]);
            if (newline_every(i) || i + 1 == n) std::cout << '\n';
        }
        break;
    }
    case RKNN_TENSOR_INT8:
    {
        const int8_t* d = static_cast<const int8_t*>(mem->virt_addr);
        for (size_t i = 0; i < n; ++i) {
            std::cout << std::setw(4) << int(d[i]);
            if (newline_every(i) || i + 1 == n) std::cout << '\n';
        }
        break;
    }
    case RKNN_TENSOR_FLOAT32:
    {
        const float* d = static_cast<const float*>(mem->virt_addr);
        for (size_t i = 0; i < n; ++i) {
            std::cout << std::setw(10) << d[i];
            if (newline_every(i) || i + 1 == n) std::cout << '\n';
        }
        break;
    }
    default:
        std::cout << "(print_tensor_raw) unsupported type\n";
    }
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

        auto* in_data = static_cast<uint8_t*>(in_mems[0]->virt_addr);

        // /* fill input[0] with const val  */
        std::fill_n(in_data, in_attr[0].size_with_stride, 10);

        // /* fill pattern [1 2 3 4] repeatedly */
        // const uint8_t pattern[4] = {1,2,3,4};
        // for (size_t off = 0; off < in_attr[0].n_elems; off += 4)
        //     std::memcpy(in_data + off, pattern,
        //                 std::min<size_t>(4, in_attr[0].n_elems - off));

        print_input_tensor(in_data, in_attr[0]);

        /* run once */
        if (rknn_run(ctx, nullptr) != RKNN_SUCC) {
            std::cerr << "rknn_run failed\n"; goto CLEANUP;
        }

        /* dump each output ― ALL VALUES */
        for (uint32_t i = 0; i < io_num.n_output; ++i) {
            std::cout << "\n── Output #" << i
                    << "  (type=" << type_str(out_attr[i].type)
                    << "  elems=" << out_attr[i].n_elems << ") ──\n";

            print_tensor_raw(out_mems[i], out_attr[i]);
        }
    }

CLEANUP:
    /* clean-up */
    for (auto* m : in_mems)  if (m) rknn_destroy_mem(ctx, m);
    for (auto* m : out_mems) if (m) rknn_destroy_mem(ctx, m);
    rknn_destroy(ctx);
    return 0;
}
