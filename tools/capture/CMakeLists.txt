cmake_minimum_required(VERSION 3.22)
project(mamba_capture LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(LLAMA_SOURCE_DIR "" CACHE PATH "Pinned llama.cpp source")
if(NOT EXISTS "${LLAMA_SOURCE_DIR}/src/llama-memory-recurrent.h")
    message(FATAL_ERROR "LLAMA_SOURCE_DIR: pinned recurrent headers required")
endif()
execute_process(COMMAND git -C "${LLAMA_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE SOURCE_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT SOURCE_COMMIT STREQUAL "8144f3192e5a3131cd043f284525e6ceebf82d0f")
    message(FATAL_ERROR "source.commit: unexpected baseline")
endif()
foreach(flag BUILD_SHARED_LIBS LLAMA_BUILD_COMMON LLAMA_BUILD_TESTS LLAMA_BUILD_TOOLS
        LLAMA_BUILD_EXAMPLES LLAMA_BUILD_SERVER LLAMA_BUILD_APP LLAMA_BUILD_UI
        LLAMA_OPENSSL GGML_BACKEND_DL GGML_BLAS GGML_ACCELERATE GGML_LLAMAFILE
        GGML_CUDA GGML_MUSA GGML_HIP GGML_VULKAN GGML_WEBGPU GGML_ZDNN
        GGML_VIRTGPU GGML_VIRTGPU_BACKEND GGML_METAL GGML_RPC GGML_SYCL
        GGML_OPENVINO GGML_ET GGML_OPENCL GGML_HEXAGON GGML_ZENDNN
        GGML_CPU_REPACK GGML_CPU_KLEIDIAI GGML_CPU_HBM GGML_OPENMP
        GGML_BUILD_TESTS GGML_BUILD_EXAMPLES)
    set(${flag} OFF CACHE BOOL "Capture CPU-only policy" FORCE)
endforeach()
set(GGML_CPU ON CACHE BOOL "Native CPU" FORCE)
add_subdirectory("${LLAMA_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/llama")
add_library(capture_core STATIC observer.cpp artifact.cpp runtime.cpp provenance.cpp)
target_include_directories(capture_core SYSTEM PUBLIC "${LLAMA_SOURCE_DIR}/src"
    "${LLAMA_SOURCE_DIR}/ggml/src" "${LLAMA_SOURCE_DIR}/vendor")
target_link_libraries(capture_core PUBLIC llama vendor::hash)
target_compile_options(capture_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
enable_testing()
add_executable(capture-test capture.test.cpp)
target_link_libraries(capture-test PRIVATE capture_core)
target_compile_options(capture-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME capture-contract COMMAND capture-test)
set_tests_properties(capture-contract PROPERTIES TIMEOUT 30
    ENVIRONMENT "GGML_CPU_DISABLE_FUSION=1;LLAMA_GRAPH_REUSE_DISABLE=1;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
add_executable(mamba-capture main.cpp)
target_link_libraries(mamba-capture PRIVATE capture_core)
target_compile_options(mamba-capture PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(capture_core PUBLIC
    CAPTURE_BUILD_DIR="${CMAKE_BINARY_DIR}" CAPTURE_TOOL_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    CAPTURE_LLAMA_SOURCE="${LLAMA_SOURCE_DIR}")
add_custom_target(capture-receipt ALL
    COMMAND ${CMAKE_COMMAND} -DTOOL_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DBUILD_DIR=${CMAKE_BINARY_DIR} -DBINARY=$<TARGET_FILE:mamba-capture>
        -DCOMPILER=${CMAKE_CXX_COMPILER} -DCOMPILER_ID=${CMAKE_CXX_COMPILER_ID}
        -DCOMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION} -P ${CMAKE_CURRENT_SOURCE_DIR}/build_identity.cmake
    DEPENDS mamba-capture VERBATIM)
