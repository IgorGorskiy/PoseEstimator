cmake_minimum_required(VERSION 3.18)  # 3.18+ нужен для CUDA language support
# Принудительно до project()
set(OpenCV_DIR "C:/vcpkg/installed/x64-windows/share/opencv4" CACHE PATH "" FORCE)
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8" CACHE PATH "" FORCE)

# В начало CMakeLists.txt после set(CUDA_TOOLKIT_ROOT_DIR...):
set(CMAKE_CUDA_COMPILER 
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe" 
    CACHE PATH "" FORCE)

set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8" CACHE PATH "" FORCE)

project(pose_estimator_gpu VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# ── CUDA архитектура ─────────────────────────────────────────────────────────
# Определяем автоматически или задаём вручную.
# RTX 20xx = 75, RTX 30xx = 86, RTX 40xx = 89
# Оставляем пустым для автоопределения (требует запуска nvcc с --run)
set(CMAKE_CUDA_ARCHITECTURES 75)

# ── OpenCV ───────────────────────────────────────────────────────────────────
# Блокируем C:/cv/opencv/build
list(REMOVE_ITEM CMAKE_PREFIX_PATH "C:/cv/opencv/build")
list(REMOVE_ITEM CMAKE_PREFIX_PATH "C:/cv/opencv")
set(OpenCV_DIR "C:/vcpkg/installed/x64-windows/share/opencv4")
# Временно отключаем FORCE-кэш vcpkg
unset(OpenCV_DIR CACHE)
set(OpenCV_DIR "C:/vcpkg/installed/x64-windows/share/opencv4" CACHE PATH "" FORCE)

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc highgui videoio calib3d)

# ── Eigen ────────────────────────────────────────────────────────────────────
find_package(Eigen3 REQUIRED NO_MODULE)

# ── OpenCASCADE ───────────────────────────────────────────────────────────────
find_package(OpenCASCADE CONFIG REQUIRED COMPONENTS
    FoundationClasses
    ModelingData
    ModelingAlgorithms
    Visualization
    ApplicationFramework
    DataExchange)

set(OCC_LIBS
    TKernel TKMath
    TKG2d TKG3d TKGeomBase TKBRep
    TKGeomAlgo TKTopAlgo TKBO TKShHealing TKHLR TKMesh
    TKXSBase TKDESTEP TKDEIGES TKDE)

# ── yaml-cpp ─────────────────────────────────────────────────────────────────
find_package(yaml-cpp REQUIRED)

# ── CUDA (уже включён через LANGUAGES выше) ───────────────────────────────────
# Заменить строку find_package(CUDAToolkit 12.8 EXACT REQUIRED) на:
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8" CACHE PATH "" FORCE)
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES 
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include" CACHE PATH "" FORCE)
find_package(CUDAToolkit REQUIRED)

# ── Sources ──────────────────────────────────────────────────────────────────
set(CXX_SOURCES
    src/main.cpp
    src/step_loader.cpp
    src/model_renderer.cpp
    src/image_preprocessor.cpp
    src/contour_matcher.cpp
    src/pose_optimizer.cpp
    src/pose_filter.cpp
    src/pipeline.cpp
)

set(CUDA_SOURCES
    src/cuda_renderer.cu
)

add_executable(${PROJECT_NAME} ${CXX_SOURCES} ${CUDA_SOURCES})

# CUDA-файлы компилируются nvcc, C++-файлы — MSVC/GCC
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCASCADE_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_calib3d
    Eigen3::Eigen
    ${OCC_LIBS}
    yaml-cpp
    CUDA::cudart
)

# ── Флаги компилятора ────────────────────────────────────────────────────────
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /wd4244 /wd4267 /wd4996>)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _USE_MATH_DEFINES NOMINMAX)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall>)
endif()

# nvcc флаги
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -arch=sm_75
    >)

# Разрешаем CUDA коду видеть Eigen (нужен --expt-relaxed-constexpr)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    EIGEN_NO_CUDA_HALF)

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

# ── Tests ────────────────────────────────────────────────────────────────────
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests)
endif()

if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND_EXPAND_LISTS)
endif()