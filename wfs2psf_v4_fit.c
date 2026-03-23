#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <float.h>
#include "common.h"

void expand_v4_double(const double *T, double *Z, long n_samples, int nx) {
    for (long i = 0; i < n_samples; i++) {
        Z[i * (nx + 1)] = 1.0;
        memcpy(&Z[i * (nx + 1) + 1], &T[i * nx], nx * sizeof(double));
    }
}

static void solve_v4_pixel(
    const double *X, const double *Y_pix,
    double *b_re, double *b_im,
    int N, int nx_eff, int iters, double lr, double lambda)
{
    // Spectral Initialization
    double *Xty = (double *)calloc(nx_eff, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < nx_eff; j++) Xty[j] += Y_pix[i] * X[i * nx_eff + j];
    }
    double norm = 1e-12;
    for (int j = 0; j < nx_eff; j++) norm += Xty[j] * Xty[j];
    norm = sqrt(norm);
    for (int j = 0; j < nx_eff; j++) {
        b_re[j] = 0.5 * Xty[j] / norm;
        b_im[j] = 0.001 * ((double)rand() / RAND_MAX - 0.5);
    }
    free(Xty);

    double *pred = (double *)malloc(N * sizeof(double));
    double *s_re = (double *)malloc(N * sizeof(double));
    double *s_im = (double *)malloc(N * sizeof(double));

    for (int iter = 0; iter < iters; iter++) {
        cblas_dgemv(CblasRowMajor, CblasNoTrans, N, nx_eff, 1.0, X, nx_eff, b_re, 1, 0.0, s_re, 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, N, nx_eff, 1.0, X, nx_eff, b_im, 1, 0.0, s_im, 1);
        for (int i = 0; i < N; i++) pred[i] = s_re[i] * s_re[i] + s_im[i] * s_im[i];

        double *g_re = (double *)calloc(nx_eff, sizeof(double));
        double *g_im = (double *)calloc(nx_eff, sizeof(double));
        for (int i = 0; i < N; i++) {
            double d = 4.0 * (pred[i] - Y_pix[i]);
            cblas_daxpy(nx_eff, d * s_re[i], &X[i * nx_eff], 1, g_re, 1);
            cblas_daxpy(nx_eff, d * s_im[i], &X[i * nx_eff], 1, g_im, 1);
        }
        for (int j = 0; j < nx_eff; j++) {
            b_re[j] -= lr * (g_re[j] / N + 2.0 * lambda * b_re[j]);
            b_im[j] -= lr * (g_im[j] / N + 2.0 * lambda * b_im[j]);
        }
        free(g_re); free(g_im);
    }
    free(pred); free(s_re); free(s_im);
}

int main(int argc, char **argv) {
    if (argc < 4) return 1;
    const char *x_f = argv[1], *y_f = argv[2], *out = argv[3];

    int nx = 10, iters = 100;
    double lr = 0.01, lambda = 1e-6;
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "-nx") == 0) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-iters") == 0) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-lr") == 0) lr = atof(argv[++i]);
        else if (strcmp(argv[i], "-lambda") == 0) lambda = atof(argv[++i]);
    }

    double *X_r, *Y_r; long N_X, P_X, N_Y, P_Y; int xa_x, ya_x, xa_y, ya_y, nax_x, nax_y;
    read_fits(x_f, &X_r, &N_X, &P_X, &xa_x, &ya_x, &nax_x);
    read_fits(y_f, &Y_r, &N_Y, &P_Y, &xa_y, &ya_y, &nax_y);

    int N = (N_X < N_Y) ? (int)N_X : (int)N_Y;

    // Center/Scale X
    double *X_m = (double *)calloc(P_X, sizeof(double));
    double *X_s = (double *)malloc(P_X * sizeof(double));
    for (int i = 0; i < N; i++) for (long j = 0; j < P_X; j++) X_m[j] += X_r[i*P_X+j];
    for (long j = 0; j < P_X; j++) X_m[j] /= N;
    for (long j = 0; j < P_X; j++) {
        double v = 0;
        for (int i = 0; i < N; i++) { double d = X_r[i*P_X+j] - X_m[j]; v += d*d; }
        X_s[j] = sqrt(v / (N - 1)) + 1e-12;
    }
    for (int i = 0; i < N; i++) for (long j = 0; j < P_X; j++) X_r[i*P_X+j] = (X_r[i * P_X + j] - X_m[j]) / X_s[j];

    // X processing (PCA or RAW)
    int nx_eff; double *Z;
    if (nx > 0) {
        printf("v4: PCA on X (%d modes)...\n", nx);
        double *Xc = (double *)malloc((long)N * P_X * sizeof(double)), *Sx = (double *)malloc(nx * sizeof(double));
        double *Ux = (double *)malloc((long)N * nx * sizeof(double)), *Vtx = (double *)malloc((long)nx * P_X * sizeof(double));
        memcpy(Xc, X_r, (long)N * P_X * sizeof(double));
        LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', N, (int)P_X, Xc, (int)P_X, Sx, Ux, nx, Vtx, (int)P_X);
        for (int i = 0; i < N; i++) for (int j = 0; j < nx; j++) Ux[i * nx + j] *= Sx[j];
        nx_eff = nx + 1; Z = (double *)malloc((long)N * nx_eff * sizeof(double));
        expand_v4_double(Ux, Z, N, nx);
        char pcp[1024]; snprintf(pcp, 1024, "%s_PCx.fits", out);
        double *PCx = (double *)malloc((long)P_X * nx * sizeof(double));
        for (long p = 0; p < P_X; p++) for (int k = 0; k < nx; k++) PCx[p * nx + k] = Vtx[k * P_X + p];
        write_fits_2d(pcp, PCx, (long)nx, (long)P_X);
        free(Xc); free(Sx); free(Ux); free(Vtx); free(PCx);
    } else {
        printf("v4: Using RAW WFS tokens...\n");
        nx = (int)P_X; nx_eff = nx + 1;
        Z = (double *)malloc((long)N * nx_eff * sizeof(double));
        expand_v4_double(X_r, Z, N, nx);
    }

    printf("Pixel-wise Field Regression for P_Y=%ld pixels...\n", P_Y);
    double *B_re = (double *)malloc((long)P_Y * nx_eff * sizeof(double));
    double *B_im = (double *)malloc((long)P_Y * nx_eff * sizeof(double));

    #pragma omp parallel for
    for (long p = 0; p < P_Y; p++) {
        double *Y_pix = (double *)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) Y_pix[i] = Y_r[i * P_Y + p];
        solve_v4_pixel(Z, Y_pix, &B_re[p * nx_eff], &B_im[p * nx_eff], N, nx_eff, iters, lr, lambda);
        free(Y_pix);
        if (p % 1000 == 0) printf("  Pixel %ld/%ld\n", p, P_Y);
    }

    char bpath[1024];
    snprintf(bpath, 1024, "%s_B_re.fits", out); write_fits_2d(bpath, B_re, (long)nx_eff, (long)P_Y);
    snprintf(bpath, 1024, "%s_B_im.fits", out); write_fits_2d(bpath, B_im, (long)nx_eff, (long)P_Y);
    snprintf(bpath, 1024, "%s_Xmean.fits", out); write_fits_2d(bpath, X_m, P_X, 1);
    snprintf(bpath, 1024, "%s_Xstd.fits", out); write_fits_2d(bpath, X_s, P_X, 1);
    snprintf(bpath, 1024, "%s_v4_info.txt", out);
    FILE *info = fopen(bpath, "w"); fprintf(info, "nx %d\nxa_y %d\nya_y %d\n", nx, xa_y, ya_y); fclose(info);
    return 0;
}
