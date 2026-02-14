cmake_minimum_required(VERSION 3.10)
project(milk-streamtelemetry-cca C)

find_package(PkgConfig REQUIRED)

pkg_check_modules(CFITSIO REQUIRED cfitsio)
pkg_check_modules(OPENBLAS REQUIRED openblas)
pkg_check_modules(LIBPNG REQUIRED libpng)
find_package(LAPACK REQUIRED)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall")

# Common utilities
add_library(common STATIC common.c)
target_include_directories(common PRIVATE ${CFITSIO_INCLUDE_DIRS})
target_link_libraries(common PRIVATE ${CFITSIO_LIBRARIES})

# CCA Program
add_executable(milk-streamtelemetry-cca cca.c)
target_include_directories(milk-streamtelemetry-cca PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
    ${OPENBLAS_INCLUDE_DIRS}
    ${LAPACKE_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-cca PRIVATE
    common
    ${CFITSIO_LIBRARIES}
    ${OPENBLAS_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    m
)

# Mode Projection Program
add_executable(milk-streamtelemetry-modeval modeval.c)
target_include_directories(milk-streamtelemetry-modeval PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
    ${OPENBLAS_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-modeval PRIVATE
    common
    ${CFITSIO_LIBRARIES}
    ${OPENBLAS_LIBRARIES}
    m
)

# Cube Shuffle Program
add_executable(milk-streamtelemetry-cubeshuffle cubeshuffle.c)
target_include_directories(milk-streamtelemetry-cubeshuffle PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-cubeshuffle PRIVATE
    common
    ${CFITSIO_LIBRARIES}
)

# PCA Program
add_executable(milk-streamtelemetry-pca pca.c)
target_include_directories(milk-streamtelemetry-pca PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
    ${OPENBLAS_INCLUDE_DIRS}
    ${LAPACKE_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-pca PRIVATE
    common
    ${CFITSIO_LIBRARIES}
    ${OPENBLAS_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    m
)

# PCA Projection Program
add_executable(milk-streamtelemetry-pcaproj pcaproj.c)
target_include_directories(milk-streamtelemetry-pcaproj PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
    ${OPENBLAS_INCLUDE_DIRS}
    ${LAPACKE_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-pcaproj PRIVATE
    common
    ${CFITSIO_LIBRARIES}
    ${OPENBLAS_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    m
)


# Reconstruction Program
add_executable(milk-streamtelemetry-recon recon.c)
target_include_directories(milk-streamtelemetry-recon PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
    ${OPENBLAS_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-recon PRIVATE
    common
    ${CFITSIO_LIBRARIES}
    ${OPENBLAS_LIBRARIES}
    m
)

# PCA Merge Program
add_executable(milk-streamtelemetry-pcamerge pcamerge.c)
target_include_directories(milk-streamtelemetry-pcamerge PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
    ${OPENBLAS_INCLUDE_DIRS}
    ${LAPACKE_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-pcamerge PRIVATE
    common
    ${CFITSIO_LIBRARIES}
    ${OPENBLAS_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    m
)

# Earth Sequence Generator
add_executable(milk-streamtelemetry-mkearthseq mkearthseq.c)
target_include_directories(milk-streamtelemetry-mkearthseq PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-mkearthseq PRIVATE
    common
    ${CFITSIO_LIBRARIES}
    m
)

# FITS 2 PNG Test Tool
add_executable(milk-streamtelemetry-3DFITS-to-png fits2png.c)
target_include_directories(milk-streamtelemetry-3DFITS-to-png PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
    ${LIBPNG_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-3DFITS-to-png PRIVATE
    common
    ${CFITSIO_LIBRARIES}
    ${LIBPNG_LIBRARIES}
    m
)

# FITS 2 ASCII Test Tool
add_executable(milk-streamtelemetry-fits2ascii fits2ascii.c)
target_include_directories(milk-streamtelemetry-fits2ascii PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-fits2ascii PRIVATE
    common
    ${CFITSIO_LIBRARIES}
    m
)

# Verify Correlation Tool
add_executable(milk-streamtelemetry-verify-corr verify_corr.c)
target_include_directories(milk-streamtelemetry-verify-corr PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-verify-corr PRIVATE
    common
    ${CFITSIO_LIBRARIES}
    m
)

install(TARGETS
    milk-streamtelemetry-cca
    milk-streamtelemetry-modeval
    milk-streamtelemetry-cubeshuffle
    milk-streamtelemetry-pca
    milk-streamtelemetry-recon
    milk-streamtelemetry-pcamerge
    milk-streamtelemetry-mkearthseq
    milk-streamtelemetry-3DFITS-to-png
    milk-streamtelemetry-fits2ascii
    milk-streamtelemetry-verify-corr
    DESTINATION /usr/local/milk/bin
)
