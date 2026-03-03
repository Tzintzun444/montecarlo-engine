cmake_minimum_required(VERSION 3.15)
project(MontecarloQuant VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(montecarlo_engine 
    bindings/bindings.cpp 
    src/montecarlo/montecarlo.cpp
)

target_include_directories(montecarlo_engine PRIVATE 
    src/montecarlo 
    src/xoshiro256
)

set_property(TARGET montecarlo_engine PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
target_link_libraries(montecarlo_engine PRIVATE OpenMP::OpenMP_CXX)

# ==========================================
# Multiplatform config
# ==========================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(montecarlo_engine PRIVATE -O3 -march=native -ffast-math -funroll-loops)
    
    if(MINGW)
        target_link_options(montecarlo_engine PRIVATE -static -static-libgcc -static-libstdc++)
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(montecarlo_engine PRIVATE /O2 /fp:fast /arch:AVX2)
endif()