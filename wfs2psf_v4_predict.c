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
    if (argc < 4) return 1;
    const char *x_f = argv[1], *mod = argv[2], *out = argv[3];

    double *X_r; long N_X, P_X; int xa_x, ya_x, nax_x;
    read_fits(x_f, &X_r, &N_X, &P_X, &xa_x, &ya_x, &nax_x);

    char path[1024]; int nx, xa_y, ya_y;
    snprintf(path, 1024, "%s_v4_info.txt", mod);
    FILE *fp = fopen(path, "r"); if(!fp) return 1;
    fscanf(fp, "nx %d\nxa_y %d\nya_y %d\n", &nx, &xa_y, &ya_y); fclose(fp);

    double *X_m, *X_s, *PCx = NULL, *B_re, *B_im;
    long n_m, p_m, n_s, p_s, n_pc, p_pc, n_bre, p_bre, n_bim, p_bim;
    int x, y, n;

    snprintf(path, 1024, "%s_Xmean.fits", mod); read_fits(path, &X_m, &n_m, &p_m, &x, &y, &n);
    snprintf(path, 1024, "%s_Xstd.fits", mod); read_fits(path, &X_s, &n_s, &p_s, &x, &y, &n);
    snprintf(path, 1024, "%s_B_re.fits", mod); read_fits(path, &B_re, &n_bre, &p_bre, &x, &y, &n);
    snprintf(path, 1024, "%s_B_im.fits", mod); read_fits(path, &B_im, &n_bim, &p_bim, &x, &y, &n);
    
    // PCx is optional (RAW mode skip)
    snprintf(path, 1024, "%s_PCx.fits", mod);
    if (access(path, F_OK) == 0) read_fits(path, &PCx, &n_pc, &p_pc, &x, &y, &n);

    // X standardization
    for (long i = 0; i < N_X; i++) for (long j = 0; j < P_X; j++) 
        X_r[i * P_X + j] = (X_r[i * P_X + j] - X_m[j]) / X_s[j];

    // nx_eff is number of columns in B_re
    int nx_eff = (int)p_bre;
    long P_Y = n_bre;

    // T = PCA projection or raw
    double *T;
    if (PCx) {
        T = (double *)malloc(N_X * nx * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N_X, nx, (int)P_X, 1.0, X_r, (int)P_X, PCx, nx, 0.0, T, nx);
    } else {
        T = X_r;
    }

    double *Z = (double *)malloc(N_X * nx_eff * sizeof(double));
    expand_v4_double(T, Z, N_X, nx);

    // Predict: S_re = Z(NxK) @ B_re^T(KxP_Y) -> (NxP_Y)
    double *S_re = (double *)malloc(N_X * P_Y * sizeof(double));
    double *S_im = (double *)malloc(N_X * P_Y * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N_X, (int)P_Y, nx_eff, 1.0, Z, nx_eff, B_re, nx_eff, 0.0, S_re, (int)P_Y);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N_X, (int)P_Y, nx_eff, 1.0, Z, nx_eff, B_im, nx_eff, 0.0, S_im, (int)P_Y);

    double *Y_new = (double *)malloc(N_X * P_Y * sizeof(double));
    for (long i = 0; i < N_X * P_Y; i++) Y_new[i] = S_re[i] * S_re[i] + S_im[i] * S_im[i];

    write_fits_3d(out, Y_new, xa_y, ya_y, (int)N_X);
    printf("v4 predict done: %s (%dx%dx%ld)\n", out, xa_y, ya_y, N_X);
    return 0;
}
