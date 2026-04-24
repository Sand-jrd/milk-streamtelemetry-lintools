cmake_minimum_required(VERSION 3.10)
project(milk-streamtelemetry-cca C)

find_package(PkgConfig REQUIRED)

pkg_check_modules(CFITSIO REQUIRED cfitsio)
pkg_check_modules(OPENBLAS REQUIRED openblas)
pkg_check_modules(LIBPNG REQUIRED libpng)
find_package(LAPACK REQUIRED)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall")

option(USE_NUMA "Use NUMA-aware allocation (Linux only)" OFF)
if(USE_NUMA)
    pkg_check_modules(NUMA numa)
    if(NUMA_FOUND)
        add_definitions(-DUSE_NUMA)
    else()
        message(WARNING "libnuma not found, disabling USE_NUMA")
        set(USE_NUMA OFF)
    endif()
endif()

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

# Iterative PCA Program
add_executable(milk-streamtelemetry-ipca ipca.c)
target_include_directories(milk-streamtelemetry-ipca PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
    ${OPENBLAS_INCLUDE_DIRS}
    ${LAPACKE_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-ipca PRIVATE
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

# PCA Projection Subtraction Program
add_executable(milk-streamtelemetry-projsub proj_sub.c)
target_include_directories(milk-streamtelemetry-projsub PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
    ${OPENBLAS_INCLUDE_DIRS}
    ${LAPACKE_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-projsub PRIVATE
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

# WFS2PSF Fit Program
add_executable(milk-streamtelemetry-wfs2psf-fit wfs2psf_fit.c)
target_include_directories(milk-streamtelemetry-wfs2psf-fit PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
    ${OPENBLAS_INCLUDE_DIRS}
    ${LAPACKE_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-wfs2psf-fit PRIVATE
    common
    ${CFITSIO_LIBRARIES}
    ${OPENBLAS_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    m
)
if(USE_NUMA)
    target_include_directories(milk-streamtelemetry-wfs2psf-fit PRIVATE ${NUMA_INCLUDE_DIRS})
    target_link_libraries(milk-streamtelemetry-wfs2psf-fit PRIVATE ${NUMA_LIBRARIES})
endif()

# WFS2PSF Predict Program
add_executable(milk-streamtelemetry-wfs2psf-predict wfs2psf_predict.c)
target_include_directories(milk-streamtelemetry-wfs2psf-predict PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
    ${OPENBLAS_INCLUDE_DIRS}
    ${LAPACKE_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-wfs2psf-predict PRIVATE
    common
    ${CFITSIO_LIBRARIES}
    ${OPENBLAS_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    m
)
if(USE_NUMA)
    target_include_directories(milk-streamtelemetry-wfs2psf-predict PRIVATE ${NUMA_INCLUDE_DIRS})
    target_link_libraries(milk-streamtelemetry-wfs2psf-predict PRIVATE ${NUMA_LIBRARIES})
endif()

# WFS2PSF Update Program
add_executable(milk-streamtelemetry-wfs2psf-update wfs2psf_v2_update.c)
target_include_directories(milk-streamtelemetry-wfs2psf-update PRIVATE
    ${CFITSIO_INCLUDE_DIRS}
    ${OPENBLAS_INCLUDE_DIRS}
    ${LAPACKE_INCLUDE_DIRS}
)
target_link_libraries(milk-streamtelemetry-wfs2psf-update PRIVATE
    common
    ${CFITSIO_LIBRARIES}
    ${OPENBLAS_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    m
)
if(USE_NUMA)
    target_include_directories(milk-streamtelemetry-wfs2psf-update PRIVATE ${NUMA_INCLUDE_DIRS})
    target_link_libraries(milk-streamtelemetry-wfs2psf-update PRIVATE ${NUMA_LIBRARIES})
endif()


install(TARGETS
    milk-streamtelemetry-cca
    milk-streamtelemetry-modeval
    milk-streamtelemetry-cubeshuffle
    milk-streamtelemetry-pca
    milk-streamtelemetry-ipca
    milk-streamtelemetry-recon
    milk-streamtelemetry-pcamerge
    milk-streamtelemetry-projsub
    milk-streamtelemetry-mkearthseq
    milk-streamtelemetry-3DFITS-to-png
    milk-streamtelemetry-fits2ascii
    milk-streamtelemetry-verify-corr
    milk-streamtelemetry-wfs2psf-fit
    milk-streamtelemetry-wfs2psf-predict
    milk-streamtelemetry-wfs2psf-update

    DESTINATION /usr/local/milk/bin
)
