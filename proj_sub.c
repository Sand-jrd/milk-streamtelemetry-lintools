#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include "common.h"

void print_help(const char *progname) {
    fprintf(stderr, "Usage: %s [options] <coeffs.fits> <modes.fits> <cube.fits> <out_cube.fits>\n", progname);
    fprintf(stderr, "\nSubtracts the PCA projection from a 3D FITS cube.\n");
    fprintf(stderr, "projection = coeffs * modes\n");
    fprintf(stderr, "out_cube = cube - projection\n");
    fprintf(stderr, "\nArguments:\n");
    fprintf(stderr, "  <coeffs.fits>   Input temporal coefficients (N, npca).\n");
    fprintf(stderr, "  <modes.fits>    Input spatial modes (npca, y, x).\n");
    fprintf(stderr, "  <cube.fits>     Input 3D FITS cube (N samples).\n");
    fprintf(stderr, "  <out_cube.fits> Output 3D FITS cube (cube - coeffs * modes).\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -float          Use single precision (float) instead of double.\n");
    fprintf(stderr, "  -h, --help      Show this help message.\n\n");
}

int main(int argc, char *argv[]) {
    int use_float = 0;
    int arg_offset = 0;

    int i = 1;
    while (i < argc) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-float") == 0) {
            use_float = 1;
            i += 1;
            arg_offset += 1;
        } else {
            break;
        }
    }

    if (argc - arg_offset < 5) {
        print_help(argv[0]);
        return 1;
    }

    const char *coeffs_file = argv[1 + arg_offset];
    const char *modes_file  = argv[2 + arg_offset];
    const char *cube_file   = argv[3 + arg_offset];
    const char *out_file    = argv[4 + arg_offset];

    if(use_float) {
        float *Coeffs = NULL;
        float *Modes = NULL;
        float *Cube = NULL;
        
        long N_coeffs, P_coeffs;
        long N_modes, P_modes;
        long N_cube, P_cube;
        
        int xa_coeffs, ya_coeffs, naxis_coeffs;
        int xa_modes, ya_modes, naxis_modes;
        int xa_cube, ya_cube, naxis_cube;

        // Read inputs
        read_fits_float(coeffs_file, &Coeffs, &N_coeffs, &P_coeffs, &xa_coeffs, &ya_coeffs, &naxis_coeffs);
        read_fits_float(modes_file, &Modes, &N_modes, &P_modes, &xa_modes, &ya_modes, &naxis_modes);
        read_fits_float(cube_file, &Cube, &N_cube, &P_cube, &xa_cube, &ya_cube, &naxis_cube);

        // Sanity checks
        if(N_coeffs != N_cube) {
            fprintf(stderr, "Error: Coeffs samples (%ld) != Cube samples (%ld)\n", N_coeffs, N_cube);
            return 1;
        }
        if(P_coeffs != N_modes) {
            fprintf(stderr, "Error: Coeffs features (%ld) != Modes count (%ld)\n", P_coeffs, N_modes);
            return 1;
        }
        if(P_modes != P_cube) {
            fprintf(stderr, "Error: Modes pixels (%ld) != Cube pixels (%ld)\n", P_modes, P_cube);
            return 1;
        }

        // Cube = Cube - Coeffs * Modes
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N_cube, P_cube, N_modes,
                    -1.0f, Coeffs, N_modes,
                    Modes, P_cube,
                    1.0f, Cube, P_cube);

        if (naxis_cube == 3) {
            write_fits_3d_float(out_file, Cube, xa_cube, ya_cube, N_cube);
        } else {
            write_fits_2d_float(out_file, Cube, xa_cube, N_cube);
        }

        free(Coeffs); free(Modes); free(Cube);

    } else {
        double *Coeffs = NULL;
        double *Modes = NULL;
        double *Cube = NULL;
        
        long N_coeffs, P_coeffs;
        long N_modes, P_modes;
        long N_cube, P_cube;
        
        int xa_coeffs, ya_coeffs, naxis_coeffs;
        int xa_modes, ya_modes, naxis_modes;
        int xa_cube, ya_cube, naxis_cube;

        // Read inputs
        read_fits(coeffs_file, &Coeffs, &N_coeffs, &P_coeffs, &xa_coeffs, &ya_coeffs, &naxis_coeffs);
        read_fits(modes_file, &Modes, &N_modes, &P_modes, &xa_modes, &ya_modes, &naxis_modes);
        read_fits(cube_file, &Cube, &N_cube, &P_cube, &xa_cube, &ya_cube, &naxis_cube);

        // Sanity checks
        if(N_coeffs != N_cube) {
            fprintf(stderr, "Error: Coeffs samples (%ld) != Cube samples (%ld)\n", N_coeffs, N_cube);
            return 1;
        }
        if(P_coeffs != N_modes) {
            fprintf(stderr, "Error: Coeffs features (%ld) != Modes count (%ld)\n", P_coeffs, N_modes);
            return 1;
        }
        if(P_modes != P_cube) {
            fprintf(stderr, "Error: Modes pixels (%ld) != Cube pixels (%ld)\n", P_modes, P_cube);
            return 1;
        }

        // Cube = Cube - Coeffs * Modes
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N_cube, P_cube, N_modes,
                    -1.0, Coeffs, N_modes,
                    Modes, P_cube,
                    1.0, Cube, P_cube);

        if (naxis_cube == 3) {
            write_fits_3d(out_file, Cube, xa_cube, ya_cube, N_cube);
        } else {
            write_fits_2d(out_file, Cube, xa_cube, N_cube);
        }

        free(Coeffs); free(Modes); free(Cube);
    }

    printf("Subtraction done: %s\n", out_file);
    return 0;
}
