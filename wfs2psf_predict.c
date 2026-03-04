#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <unistd.h>
#include "common.h"

void quadratic_expand_double(const double *T, double *Z, long n_samples, int nx) {
    int z_dim = nx + nx*(nx+1)/2;
    for (long i = 0; i < n_samples; i++) {
        const double *t = &T[i * nx];
        double *z = &Z[i * z_dim];
        int idx = 0;
        for (int j = 0; j < nx; j++) z[idx++] = t[j];
        for (int j = 0; j < nx; j++) {
            for (int k = j; k < nx; k++) {
                z[idx++] = t[j] * t[k];
            }
        }
    }
}

void quadratic_expand_float(const float *T, float *Z, long n_samples, int nx) {
    int z_dim = nx + nx*(nx+1)/2;
    for (long i = 0; i < n_samples; i++) {
        const float *t = &T[i * nx];
        float *z = &Z[i * z_dim];
        int idx = 0;
        for (int j = 0; j < nx; j++) z[idx++] = t[j];
        for (int j = 0; j < nx; j++) {
            for (int k = j; k < nx; k++) {
                z[idx++] = t[j] * t[k];
            }
        }
    }
}

void print_help(const char *prog) {
    printf("Usage: %s <X_new.fits> <model_prefix> <out_Y.fits> [options]\n", prog);
    printf("Options:\n");
    printf("  -noscale              Disable scale (default: enabled if std files exist)\n");
    printf("  -noquad               Disable quadratic expansion (default: enabled)\n");
    printf("  -float                Use single precision (default is double)\n");
}

int main(int argc, char **argv) {
    if (argc < 4) {
        print_help(argv[0]);
        return 1;
    }

    const char *x_file = argv[1];
    const char *model_prefix = argv[2];
    const char *out_file = argv[3];

    int scale = 1;
    int use_quadratic = 1;
    int use_float = 0;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "-noscale") == 0) scale = 0;
        else if (strcmp(argv[i], "-noquad") == 0) use_quadratic = 0;
        else if (strcmp(argv[i], "-float") == 0) use_float = 1;
    }

    // Check if PCy exists
    char pcy_path[1024];
    snprintf(pcy_path, 1024, "%s_PCy.fits", model_prefix);
    int y_pca_mode = (access(pcy_path, F_OK) == 0);

    if (!use_float) {
        double *X = NULL;
        long N, P_X;
        int xa_x, ya_x, nax_x;
        read_fits(x_file, &X, &N, &P_X, &xa_x, &ya_x, &nax_x);

        char path[1024];
        double *X_mean, *X_std, *Y_mean, *Y_std, *PCx, *B_latent, *PCy = NULL;
        long n1, p1, nx, p2;
        int xa, ya, nax;

        snprintf(path, 1024, "%s_Xmean.fits", model_prefix);
        read_fits(path, &X_mean, &n1, &p1, &xa, &ya, &nax);

        snprintf(path, 1024, "%s_Xstd.fits", model_prefix);
        read_fits(path, &X_std, &n1, &p1, &xa, &ya, &nax);

        snprintf(path, 1024, "%s_Ymean.fits", model_prefix);
        read_fits(path, &Y_mean, &n1, &p2, &xa, &ya, &nax);
        long P_Y = n1 * p2; 

        snprintf(path, 1024, "%s_Ystd.fits", model_prefix);
        read_fits(path, &Y_std, &n1, &p2, &xa, &ya, &nax);

        snprintf(path, 1024, "%s_PCx.fits", model_prefix);
        read_fits(path, &PCx, &n1, &nx, &xa, &ya, &nax);

        snprintf(path, 1024, "%s_B_latent.fits", model_prefix);
        long z_dim, target_dim;
        read_fits(path, &B_latent, &z_dim, &target_dim, &xa, &ya, &nax);
        
        long ny = 0;
        if (y_pca_mode) {
            snprintf(path, 1024, "%s_PCy.fits", model_prefix);
            read_fits(path, &PCy, &n1, &ny, &xa, &ya, &nax);
        }

        // Center and scale X
        for (long i = 0; i < N; i++) {
            for (long j = 0; j < P_X; j++) {
                X[i * P_X + j] -= X_mean[j];
                if (scale) X[i * P_X + j] /= X_std[j];
            }
        }

        // T_new = Xc_new @ PCx
        double *T = (double *)malloc(N * nx * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, nx, P_X, 1.0, X, P_X, PCx, nx, 0.0, T, nx);

        // Quadratic latent regression
        double *Z = T;
        if (use_quadratic) {
            Z = (double *)malloc(N * z_dim * sizeof(double));
            quadratic_expand_double(T, Z, N, nx);
        }

        double *U_pred = (double *)malloc(N * target_dim * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, target_dim, z_dim, 1.0, Z, z_dim, B_latent, target_dim, 0.0, U_pred, target_dim);

        double *Y_new;
        if (y_pca_mode) {
            Y_new = (double *)malloc(N * P_Y * sizeof(double));
            // Y_new = U_pred @ PCy.T
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        N, P_Y, ny, 1.0, U_pred, ny, PCy, ny, 0.0, Y_new, P_Y);
        } else {
            Y_new = U_pred;
        }

        if (scale) {
            for (long i = 0; i < N; i++) {
                for (long j = 0; j < P_Y; j++) {
                    Y_new[i * P_Y + j] = Y_new[i * P_Y + j] * Y_std[j] + Y_mean[j];
                }
            }
        } else {
            for (long i = 0; i < N; i++) {
                for (long j = 0; j < P_Y; j++) {
                    Y_new[i * P_Y + j] += Y_mean[j]; // In python it scales if self.scale, but mean is always added back since Yc = Y - Y_mean
                }
            }
        }

        // write Y_new
        // if nax_x == 3, maybe we want to write a 3D FITS if Y is a cube.
        // Actually Y could be 2D. We should save as 2D (P_Y, N) where P_Y is columns, N is rows
        write_fits_2d(out_file, Y_new, P_Y, N);

        free(X); free(X_mean); free(X_std); free(Y_mean); free(Y_std);
        free(PCx); free(B_latent);
        if (PCy) free(PCy);
        free(T);
        if (use_quadratic) free(Z);
        if (y_pca_mode) free(Y_new);
        if (y_pca_mode) free(U_pred); // U_pred is Y_new if not y_pca_mode
    } else {
        // ===================================
        // SINGLE PRECISION PATH
        // ===================================
        float *X = NULL;
        long N, P_X;
        int xa_x, ya_x, nax_x;
        read_fits_float(x_file, &X, &N, &P_X, &xa_x, &ya_x, &nax_x);

        char path[1024];
        float *X_mean, *X_std, *Y_mean, *Y_std, *PCx, *B_latent, *PCy = NULL;
        long n1, p1, nx, p2;
        int xa, ya, nax;

        snprintf(path, 1024, "%s_Xmean.fits", model_prefix);
        read_fits_float(path, &X_mean, &n1, &p1, &xa, &ya, &nax);

        snprintf(path, 1024, "%s_Xstd.fits", model_prefix);
        read_fits_float(path, &X_std, &n1, &p1, &xa, &ya, &nax);

        snprintf(path, 1024, "%s_Ymean.fits", model_prefix);
        read_fits_float(path, &Y_mean, &n1, &p2, &xa, &ya, &nax);
        long P_Y = n1 * p2; 

        snprintf(path, 1024, "%s_Ystd.fits", model_prefix);
        read_fits_float(path, &Y_std, &n1, &p2, &xa, &ya, &nax);

        snprintf(path, 1024, "%s_PCx.fits", model_prefix);
        read_fits_float(path, &PCx, &n1, &nx, &xa, &ya, &nax);

        snprintf(path, 1024, "%s_B_latent.fits", model_prefix);
        long z_dim, target_dim;
        read_fits_float(path, &B_latent, &z_dim, &target_dim, &xa, &ya, &nax);
        
        long ny = 0;
        if (y_pca_mode) {
            snprintf(path, 1024, "%s_PCy.fits", model_prefix);
            read_fits_float(path, &PCy, &n1, &ny, &xa, &ya, &nax);
        }

        for (long i = 0; i < N; i++) {
            for (long j = 0; j < P_X; j++) {
                X[i * P_X + j] -= X_mean[j];
                if (scale) X[i * P_X + j] /= X_std[j];
            }
        }

        float *T = (float *)malloc(N * nx * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, nx, P_X, 1.0f, X, P_X, PCx, nx, 0.0f, T, nx);

        float *Z = T;
        if (use_quadratic) {
            Z = (float *)malloc(N * z_dim * sizeof(float));
            quadratic_expand_float(T, Z, N, nx);
        }

        float *U_pred = (float *)malloc(N * target_dim * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, target_dim, z_dim, 1.0f, Z, z_dim, B_latent, target_dim, 0.0f, U_pred, target_dim);

        float *Y_new;
        if (y_pca_mode) {
            Y_new = (float *)malloc(N * P_Y * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        N, P_Y, ny, 1.0f, U_pred, ny, PCy, ny, 0.0f, Y_new, P_Y);
        } else {
            Y_new = U_pred;
        }

        if (scale) {
            for (long i = 0; i < N; i++) {
                for (long j = 0; j < P_Y; j++) {
                    Y_new[i * P_Y + j] = Y_new[i * P_Y + j] * Y_std[j] + Y_mean[j];
                }
            }
        } else {
            for (long i = 0; i < N; i++) {
                for (long j = 0; j < P_Y; j++) {
                    Y_new[i * P_Y + j] += Y_mean[j];
                }
            }
        }

        write_fits_2d_float(out_file, Y_new, P_Y, N);

        free(X); free(X_mean); free(X_std); free(Y_mean); free(Y_std);
        free(PCx); free(B_latent);
        if (PCy) free(PCy);
        free(T);
        if (use_quadratic) free(Z);
        if (y_pca_mode) free(Y_new);
        if (y_pca_mode) free(U_pred);
    }

    printf("Done. Saved %s\n", out_file);
    return 0;
}
