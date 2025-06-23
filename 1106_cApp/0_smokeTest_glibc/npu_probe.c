// npu_probe.c
#include "rknn_api.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
    const char* model_path = (argc > 1) ? argv[1] : NULL;

    printf("╔═══════════════════════════════════════════════\n");
    printf("║ NPU probe test\n");
    printf("╠═══════════════════════════════════════════════\n");
    if (model_path)
        printf("║ Loading model: %s\n", model_path);
    else
        printf("║ No model supplied – calling rknn_init(NULL)…\n");

    rknn_context ctx;
    int ret = rknn_init(&ctx,
                        model_path,      /* model buffer or path        */
                        0,               /* size (0 → treat as file)     */
                        0,               /* flags (0 = default)          */
                        NULL);           /* rknn_context_attr (unused)   */

    printf("╚→ rknn_init returned %d\n", ret);

    if (ret == 0)
        rknn_destroy(ctx);               /* tidy-up when a model loaded */

    return (ret == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
