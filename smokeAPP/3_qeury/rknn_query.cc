#include <iostream>
#include <rknn_api.h>
#include <cstring>
#include <cstdlib>

const char* tensor_fmt(int fmt) {
    switch (fmt) {
        case RKNN_TENSOR_NCHW: return "NCHW";
        case RKNN_TENSOR_NHWC: return "NHWC";
        default: return "UNKNOWN_FMT";
    }
}

const char* tensor_type(int type) {
    switch (type) {
        case RKNN_TENSOR_FLOAT32: return "float32";
        case RKNN_TENSOR_INT8:    return "int8";
        case RKNN_TENSOR_UINT8:   return "uint8";
        default: return "UNKNOWN_TYPE";
    }
}

const char* qnt_type_name(int q) {
    switch (q) {
        case RKNN_TENSOR_QNT_NONE:                return "NONE";
        case RKNN_TENSOR_QNT_DFP:                 return "DFP";
        case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC:   return "ASYM (u8/i8)";
        default: return "UNKNOWN_OR_UNSUPPORTED_QNT_TYPE";
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " model.rknn\n";
        return 1;
    }

    const char* model_path = argv[1];
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        std::cerr << "Failed to open model: " << model_path << "\n";
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    rewind(fp);
    void* model_data = malloc(size);
    fread(model_data, 1, size, fp);
    fclose(fp);

    rknn_context ctx;
    if (rknn_init(&ctx, model_data, size, 0, nullptr) != 0) {
        std::cerr << "rknn_init failed\n";
        return 1;
    }

    // Version info
    rknn_sdk_version ver;
    if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == 0) {
        std::cout << "RKNN SDK Version: " << ver.api_version
                  << ", Driver Version: " << ver.drv_version << "\n";
    }

    // Query IO count
    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    std::cout << "\nModel I/O: " << io_num.n_input << " input(s), "
              << io_num.n_output << " output(s)\n";

    // Input attributes
    for (int i = 0; i < io_num.n_input; ++i) {
        rknn_tensor_attr attr = {};
        attr.index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));

        std::cout << "\n[Input " << i << "]\n";
        std::cout << "  Name       : " << attr.name << "\n";
        std::cout << "  Dims       : [" << attr.dims[0] << ", "
                                      << attr.dims[1] << ", "
                                      << attr.dims[2] << ", "
                                      << attr.dims[3] << "]\n";
        std::cout << "  Type       : " << tensor_type(attr.type) << "\n";
        std::cout << "  Qnt Type   : " << qnt_type_name(attr.qnt_type) << "\n";
        std::cout << "  Format     : " << tensor_fmt(attr.fmt) << "\n";

        int count = 1;
        for (int j = 0; j < attr.n_dims; ++j)
            count *= attr.dims[j];

        int bytes_per_elem = (attr.type == RKNN_TENSOR_INT8 || attr.type == RKNN_TENSOR_UINT8) ? 1 : 4;

        std::cout << "  Elements   : " << count << "\n";
        std::cout << "  Buffer Size: " << count * bytes_per_elem << " bytes\n";

        std::cout << "  📦 To fill input buffer:\n"
                  << "    - Allocate array of " << count << " x "
                  << tensor_type(attr.type) << "\n"
                  << "    - Layout: " << tensor_fmt(attr.fmt)
                  << " shape [" << attr.dims[0] << ", "
                                << attr.dims[1] << ", "
                                << attr.dims[2] << ", "
                                << attr.dims[3] << "]\n";
    }

    // Output attributes
    for (int i = 0; i < io_num.n_output; ++i) {
        rknn_tensor_attr attr = {};
        attr.index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));

        std::cout << "\n[Output " << i << "]\n";
        std::cout << "  Name       : " << attr.name << "\n";
        std::cout << "  Dims       : [" << attr.dims[0] << ", "
                                      << attr.dims[1] << ", "
                                      << attr.dims[2] << ", "
                                      << attr.dims[3] << "]\n";
        std::cout << "  Type       : " << tensor_type(attr.type) << "\n";
        std::cout << "  Qnt Type   : " << qnt_type_name(attr.qnt_type) << "\n";
        std::cout << "  Format     : " << tensor_fmt(attr.fmt) << "\n";

        int count = 1;
        for (int j = 0; j < attr.n_dims; ++j)
            count *= attr.dims[j];

        int bytes_per_elem = (attr.type == RKNN_TENSOR_INT8 || attr.type == RKNN_TENSOR_UINT8) ? 1 : 4;

        std::cout << "  Elements   : " << count << "\n";
        std::cout << "  Output Size: " << count * bytes_per_elem << " bytes\n";
    }

    rknn_destroy(ctx);
    free(model_data);
    return 0;
}
