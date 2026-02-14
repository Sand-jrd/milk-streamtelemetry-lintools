#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include "common.h"

void print_help(const char *progname) {
    fprintf(stderr, "Usage: %s [options] <input.fits> <modes.fits> <coeffs.fits>\n", progname);
    fprintf(stderr, "\nProjects a new 3D FITS cube onto precomputed PCA modes.\n");
}

int main(int argc, char *argv[]) {
    if(argc < 4) {
        print_help(argv[0]);
        return 1;
    }

    const char *infile      = argv[1];
    const char *modes_file  = argv[2];
    const char *coeffs_file = argv[3];

    int use_float = 0; // set to 1 for float precision, 0 for double

    if(use_float) {
        float *X = NULL;
        float *Modes = NULL;
        long N_cube, P_cube;
        long N_modes, P_modes;
        int xa_cube, ya_cube, naxis_cube;
        int xa_modes, ya_modes, naxis_modes;

        // Read input cube
        read_fits_float(infile, &X, &N_cube, &P_cube, &xa_cube, &ya_cube, &naxis_cube);
        // Read precomputed modes
        read_fits_float(modes_file, &Modes, &N_modes, &P_modes, &xa_modes, &ya_modes, &naxis_modes);

        // Check that spatial dimensions match
        if(P_cube != P_modes) {
            fprintf(stderr, "Error: Cube pixels (%ld) != Modes pixels (%ld)\n", P_cube, P_modes);
            free(X); free(Modes);
            return 1;
        }

        // Allocate Coeffs array (N_cube x N_modes)
        float *Coeffs = (float *)malloc(N_cube * N_modes * sizeof(float));
        if(!Coeffs) {
            fprintf(stderr, "Error: Failed to allocate memory for Coeffs\n");
            free(X); free(Modes);
            return 1;
        }

        // Compute projection: Coeffs = X * Modes^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N_cube, N_modes, P_cube,
                    1.0f, X, P_cube,
                    Modes, P_modes,
                    0.0f, Coeffs, N_modes);

        // Write output coefficients
        write_fits_2d_float(coeffs_file, Coeffs, N_modes, N_cube);

        free(X); free(Modes); free(Coeffs);

    } else {
        double *X = NULL;
        double *Modes = NULL;
        long N_cube, P_cube;
        long N_modes, P_modes;
        int xa_cube, ya_cube, naxis_cube;
        int xa_modes, ya_modes, naxis_modes;

        // Read input cube (double)
        read_fits(infile, &X, &N_cube, &P_cube, &xa_cube, &ya_cube, &naxis_cube);
        // Read modes (double)
        read_fits(modes_file, &Modes, &N_modes, &P_modes, &xa_modes, &ya_modes, &naxis_modes);

        // Check spatial dimensions
        if(P_cube != P_modes) {
            fprintf(stderr, "Error: Cube pixels (%ld) != Modes pixels (%ld)\n", P_cube, P_modes);
            free(X); free(Modes);
            return 1;
        }

        // Allocate Coeffs array
        double *Coeffs = (double *)malloc(N_cube * N_modes * sizeof(double));
        if(!Coeffs) {
            fprintf(stderr, "Error: Failed to allocate memory for Coeffs\n");
            free(X); free(Modes);
            return 1;
        }

        // Project: Coeffs = X * Modes^T
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N_cube, N_modes, P_cube,
                    1.0, X, P_cube,
                    Modes, P_modes,
                    0.0, Coeffs, N_modes);

        // Write output coefficients
        write_fits_2d(coeffs_file, Coeffs, N_modes, N_cube);

        free(X); free(Modes); free(Coeffs);
    }

    printf("Projection done: %s\n", coeffs_file);
    return 0;
}
