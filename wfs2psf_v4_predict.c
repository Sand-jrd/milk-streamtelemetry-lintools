#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <unistd.h>
#include "common.h"

void expand_v4_double(const double *T, double *Z, long n_samples, int nx) {
    for (long i = 0; i < n_samples; i++) {
        Z[i * (nx + 1)] = 1.0;
        memcpy(&Z[i * (nx + 1) + 1], &T[i * nx], nx * sizeof(double));
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s <X_new.fits> <model_prefix> <out_Y.fits>\n", argv[0]);
        return 1;
    }

    const char *x_file = argv[1];
    const char *model_prefix = argv[2];
    const char *out_file = argv[3];

    double *X = NULL;
    long N, P_X;
    int xa_x, ya_x, nax_x;
    read_fits(x_file, &X, &N, &P_X, &xa_x, &ya_x, &nax_x);

    char path[1024];
    double *X_mean, *X_std, *PCx, *B_re, *B_im;
    long n1, p1, n_pix;
    int xa, ya, nax, nx;

    snprintf(path, 1024, "%s_v4_info.txt", model_prefix);
    int patchsize, ny_per_patch, nxp, nyp;
    FILE *fp = fopen(path, "r");
    if(!fp) { fprintf(stderr, "Model info not found.\n"); return 1; }
    fscanf(fp, "patchsize %d\nny_per_patch %d\nnxp %d\nnyp %d\nnx %d\n", &patchsize, &ny_per_patch, &nxp, &nyp, &nx);
    fclose(fp);

    snprintf(path, 1024, "%s_Xmean.fits", model_prefix);
    read_fits(path, &X_mean, &n1, &p1, &xa, &ya, &nax);
    snprintf(path, 1024, "%s_Xstd.fits", model_prefix);
    read_fits(path, &X_std, &n1, &p1, &xa, &ya, &nax);
    snprintf(path, 1024, "%s_PCx.fits", model_prefix);
    read_fits(path, &PCx, &n1, &p1, &xa, &ya, &nax);
    snprintf(path, 1024, "%s_B_re.fits", model_prefix);
    long nx_eff_long;
    read_fits(path, &B_re, &nx_eff_long, &n_pix, &xa, &ya, &nax);
    snprintf(path, 1024, "%s_B_im.fits", model_prefix);
    read_fits(path, &B_im, &nx_eff_long, &n_pix, &xa, &ya, &nax);

    int nx_eff = (int)nx_eff_long;

    // Preprocess X
    for (long i = 0; i < N; i++) {
        for (long j = 0; j < P_X; j++) {
            X[i * P_X + j] = (X[i * P_X + j] - X_mean[j]) / X_std[j];
        }
    }

    // T = X @ PCx
    double *T = (double *)malloc(N * nx * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N, nx, (int)P_X, 1.0, X, (int)P_X, PCx, nx, 0.0, T, nx);

    // Z = [1.0, T]
    double *Z = (double *)malloc(N * nx_eff * sizeof(double));
    expand_v4_double(T, Z, N, nx);

    // Predict: Y = (Z @ B_re^T)^2 + (Z @ B_im^T)^2
    // B_re is n_pix x nx_eff -> we need Z(N x nx_eff) @ B_re^T (nx_eff x n_pix)
    double *S_re = (double *)malloc(N * n_pix * sizeof(double));
    double *S_im = (double *)malloc(N * n_pix * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N, (int)n_pix, nx_eff, 1.0, Z, nx_eff, B_re, nx_eff, 0.0, S_re, (int)n_pix);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, (int)N, (int)n_pix, nx_eff, 1.0, Z, nx_eff, B_im, nx_eff, 0.0, S_im, (int)n_pix); // Wait B is n_pix x nx_eff

    // Wait above dgemm for S_im has CblasTrans for Z? No.
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N, (int)n_pix, nx_eff, 1.0, Z, nx_eff, B_im, nx_eff, 0.0, S_im, (int)n_pix);

    double *Y_new = (double *)malloc(N * n_pix * sizeof(double));
    for (long i = 0; i < N * n_pix; i++) {
        Y_new[i] = S_re[i] * S_re[i] + S_im[i] * S_im[i];
    }

    // Note: n_pix here matches the total pixels in patches. 
    // We should reshape it back to the original ya_y * xa_y if needed.
    // For now we save as 2D (concatenated patches).
    write_fits_2d(out_file, Y_new, n_pix, N);

    printf("v4 predict done. Saved %s\n", out_file);
    return 0;
}
