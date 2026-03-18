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

    if (!use_float) {
        double *X = NULL;
        long N, P_X;
        int xa_x, ya_x, nax_x;
        read_fits(x_file, &X, &N, &P_X, &xa_x, &ya_x, &nax_x);

        char path[1024];
        double *X_mean, *X_std, *Y_mean, *Y_std, *PCx, *B_latent, *PCy_local = NULL;
        long n1, p1, nx, p2;
        int xa, ya, nax;

        snprintf(path, 1024, "%s_Xmean.fits", model_prefix);
        read_fits(path, &X_mean, &n1, &p1, &xa, &ya, &nax);
        snprintf(path, 1024, "%s_Xstd.fits", model_prefix);
        read_fits(path, &X_std, &n1, &p1, &xa, &ya, &nax);
        snprintf(path, 1024, "%s_Ymean.fits", model_prefix);
        int xa_y, ya_y, nax_y;
        read_fits(path, &Y_mean, &n1, &p2, &xa_y, &ya_y, &nax_y);
        long P_Y = n1 * p2; 
        snprintf(path, 1024, "%s_Ystd.fits", model_prefix);
        read_fits(path, &Y_std, &n1, &p2, &xa, &ya, &nax);
        snprintf(path, 1024, "%s_PCx.fits", model_prefix);
        read_fits(path, &PCx, &n1, &nx, &xa, &ya, &nax);
        snprintf(path, 1024, "%s_B_latent.fits", model_prefix);
        long z_dim, target_dim;
        read_fits(path, &B_latent, &z_dim, &target_dim, &xa, &ya, &nax);

        // Load v2 info
        int patchsize, ny_per_patch, nxp, nyp, y_pca_mode = 1;
        snprintf(path, 1024, "%s_v2_info.txt", model_prefix);
        FILE *fp = fopen(path, "r");
        if (!fp) { fprintf(stderr, "Error: %s not found. Model might not be v2.\n", path); exit(1); }
        fscanf(fp, "patchsize %d\n", &patchsize);
        fscanf(fp, "ny_per_patch %d\n", &ny_per_patch);
        fscanf(fp, "nxp %d\n", &nxp);
        fscanf(fp, "nyp %d\n", &nyp);
        if (fscanf(fp, "y_pca_mode %d\n", &y_pca_mode) != 1) y_pca_mode = 1; // Default to 1 for older v2 models
        fclose(fp);

        if (y_pca_mode) {
            snprintf(path, 1024, "%s_PCy_local.fits", model_prefix);
            long n_p_l, p_dim_l;
            read_fits(path, &PCy_local, &n_p_l, &p_dim_l, &xa, &ya, &nax);
        }

        for (long i = 0; i < N; i++) {
            for (long j = 0; j < P_X; j++) {
                X[i * P_X + j] -= X_mean[j];
                if (scale) X[i * P_X + j] /= X_std[j];
            }
        }

        double *T = (double *)malloc(N * nx * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N, (int)nx, (int)P_X, 1.0, X, (int)P_X, PCx, (int)nx, 0.0, T, (int)nx);

        double *Z = T;
        if (use_quadratic) {
            Z = (double *)malloc(N * z_dim * sizeof(double));
            quadratic_expand_double(T, Z, N, (int)nx);
        }

        double *U_pred = (double *)malloc(N * target_dim * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N, (int)target_dim, (int)z_dim, 1.0, Z, (int)z_dim, B_latent, (int)target_dim, 0.0, U_pred, (int)target_dim);

        double *Y_new = (double *)calloc(N * P_Y, sizeof(double));
        int patch_pixels = patchsize * patchsize;
        for (int i = 0; i < N; i++) {
            for (int py = 0; py < nyp; py++) {
                for (int px = 0; px < nxp; px++) {
                    int patch_idx = py * nxp + px;
                    double *U_patch = &U_pred[i * target_dim + patch_idx * ny_per_patch];
                    if (y_pca_mode) {
                        double *PCy_patch = &PCy_local[patch_idx * (patch_pixels * ny_per_patch)];
                        for (int dy = 0; dy < patchsize; dy++) {
                            for (int dx = 0; dx < patchsize; dx++) {
                                double val = 0;
                                int pixel_idx = dy * patchsize + dx;
                                for (int k = 0; k < ny_per_patch; k++) {
                                    val += U_patch[k] * PCy_patch[pixel_idx * ny_per_patch + k];
                                }
                                Y_new[i * P_Y + (py * patchsize + dy) * xa_y + (px * patchsize + dx)] = val;
                            }
                        }
                    } else {
                        // Raw pixels
                        for (int k = 0; k < patch_pixels; k++) {
                            int dy = k / patchsize;
                            int dx = k % patchsize;
                            Y_new[i * P_Y + (py * patchsize + dy) * xa_y + (px * patchsize + dx)] = U_patch[k];
                        }
                    }
                }
            }
        }

        for (long i = 0; i < N; i++) {
            for (long j = 0; j < P_Y; j++) {
                Y_new[i * P_Y + j] = Y_new[i * P_Y + j] * Y_std[j] + Y_mean[j];
            }
        }

        if (nax_y == 3) write_fits_3d(out_file, Y_new, xa_y, ya_y, (int)N);
        else write_fits_2d(out_file, Y_new, (int)P_Y, (int)N);

        free(X); free(X_mean); free(X_std); free(Y_mean); free(Y_std);
        free(PCx); free(B_latent); free(PCy_local); free(T); 
        if (use_quadratic) free(Z);
        free(Y_new); free(U_pred);
    } else {
        // ===================================
        // SINGLE PRECISION PATH
        // ===================================
        float *X = NULL;
        long N, P_X;
        int xa_x, ya_x, nax_x;
        read_fits_float(x_file, &X, &N, &P_X, &xa_x, &ya_x, &nax_x);

        char path[1024];
        float *X_mean, *X_std, *Y_mean, *Y_std, *PCx, *B_latent, *PCy_local = NULL;
        long n1, p1, nx, p2;
        int xa, ya, nax;

        snprintf(path, 1024, "%s_Xmean.fits", model_prefix);
        read_fits_float(path, &X_mean, &n1, &p1, &xa, &ya, &nax);
        snprintf(path, 1024, "%s_Xstd.fits", model_prefix);
        read_fits_float(path, &X_std, &n1, &p1, &xa, &ya, &nax);
        snprintf(path, 1024, "%s_Ymean.fits", model_prefix);
        int xa_y, ya_y, nax_y;
        read_fits_float(path, &Y_mean, &n1, &p2, &xa_y, &ya_y, &nax_y);
        long P_Y = n1 * p2; 
        snprintf(path, 1024, "%s_Ystd.fits", model_prefix);
        read_fits_float(path, &Y_std, &n1, &p2, &xa, &ya, &nax);
        snprintf(path, 1024, "%s_PCx.fits", model_prefix);
        read_fits_float(path, &PCx, &n1, &nx, &xa, &ya, &nax);
        snprintf(path, 1024, "%s_B_latent.fits", model_prefix);
        long z_dim, target_dim;
        read_fits_float(path, &B_latent, &z_dim, &target_dim, &xa, &ya, &nax);

        // Load v2 info
        int patchsize, ny_per_patch, nxp, nyp, y_pca_mode = 1;
        snprintf(path, 1024, "%s_v2_info.txt", model_prefix);
        FILE *fp = fopen(path, "r");
        if (!fp) { fprintf(stderr, "Error: %s not found. Model might not be v2.\n", path); exit(1); }
        fscanf(fp, "patchsize %d\n", &patchsize);
        fscanf(fp, "ny_per_patch %d\n", &ny_per_patch);
        fscanf(fp, "nxp %d\n", &nxp);
        fscanf(fp, "nyp %d\n", &nyp);
        if (fscanf(fp, "y_pca_mode %d\n", &y_pca_mode) != 1) y_pca_mode = 1;
        fclose(fp);

        if (y_pca_mode) {
            snprintf(path, 1024, "%s_PCy_local.fits", model_prefix);
            long n_p_l, p_dim_l;
            read_fits_float(path, &PCy_local, &n_p_l, &p_dim_l, &xa, &ya, &nax);
        }

        for (long i = 0; i < N; i++) {
            for (long j = 0; j < P_X; j++) {
                X[i * P_X + j] -= X_mean[j];
                if (scale) X[i * P_X + j] /= X_std[j];
            }
        }

        float *T = (float *)malloc(N * nx * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N, (int)nx, (int)P_X, 1.0f, X, (int)P_X, PCx, (int)nx, 0.0f, T, (int)nx);

        float *Z = T;
        if (use_quadratic) {
            Z = (float *)malloc(N * z_dim * sizeof(float));
            quadratic_expand_float(T, Z, N, (int)nx);
        }

        float *U_pred = (float *)malloc(N * target_dim * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N, (int)target_dim, (int)z_dim, 1.0f, Z, (int)z_dim, B_latent, (int)target_dim, 0.0f, U_pred, (int)target_dim);

        float *Y_new = (float *)calloc(N * P_Y, sizeof(float));
        int patch_pixels = patchsize * patchsize;
        for (int i = 0; i < N; i++) {
            for (int py = 0; py < nyp; py++) {
                for (int px = 0; px < nxp; px++) {
                    int patch_idx = py * nxp + px;
                    float *U_patch = &U_pred[i * target_dim + patch_idx * ny_per_patch];
                    if (y_pca_mode) {
                        float *PCy_patch = &PCy_local[patch_idx * (patch_pixels * ny_per_patch)];
                        for (int dy = 0; dy < patchsize; dy++) {
                            for (int dx = 0; dx < patchsize; dx++) {
                                float val = 0;
                                int pixel_idx = dy * patchsize + dx;
                                for (int k = 0; k < ny_per_patch; k++) {
                                    val += U_patch[k] * PCy_patch[pixel_idx * ny_per_patch + k];
                                }
                                Y_new[i * P_Y + (py * patchsize + dy) * xa_y + (px * patchsize + dx)] = val;
                            }
                        }
                    } else {
                        for (int k = 0; k < patch_pixels; k++) {
                            int dy = k / patchsize;
                            int dx = k % patchsize;
                            Y_new[i * P_Y + (py * patchsize + dy) * xa_y + (px * patchsize + dx)] = U_patch[k];
                        }
                    }
                }
            }
        }

        for (long i = 0; i < N; i++) {
            for (long j = 0; j < P_Y; j++) {
                Y_new[i * P_Y + j] = Y_new[i * P_Y + j] * Y_std[j] + Y_mean[j];
            }
        }

        if (nax_y == 3) write_fits_3d_float(out_file, Y_new, xa_y, ya_y, (int)N);
        else write_fits_2d_float(out_file, Y_new, (int)P_Y, (int)N);

        free(X); free(X_mean); free(X_std); free(Y_mean); free(Y_std);
        free(PCx); free(B_latent); free(PCy_local); free(T); 
        if (use_quadratic) free(Z);
        free(Y_new); free(U_pred);
    }

    printf("Done. Saved %s\n", out_file);
    return 0;
}
