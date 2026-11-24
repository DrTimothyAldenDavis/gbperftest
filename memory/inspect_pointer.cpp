void inspect_pointer(const void* ptr) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

    if (err != cudaSuccess) {
        // Older CUDA versions returned cudaErrorInvalidValue for unregistered host pointers
        cudaGetLastError(); // clear sticky error
        printf("Unregistered host pointer (CPU)\n");
        return;
    }

    switch (attr.type) {
        case cudaMemoryTypeUnregistered:
            printf("Location: Unregistered CPU memory (malloc / new / stack)\n");
            break;
        case cudaMemoryTypeHost:
            printf("Location: Registered / pinned CPU memory (cudaMallocHost)\n");
            break;
        case cudaMemoryTypeDevice:
            printf("Location: GPU Device %d\n", attr.device);
            break;
        case cudaMemoryTypeManaged:
            printf("Location: Unified / Managed memory (preferred/associated device: %d)\n", attr.device);
            break;
    }
}
