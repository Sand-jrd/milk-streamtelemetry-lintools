#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "cblas.h"
#include "lapacke.h"
#include <fitsio.h>
#include <omp.h>

#ifdef USE_NUMA
#include <numa.h>
#define malloc_numa(s) numa_alloc_interleaved(s)
#define calloc_numa(n, s) numa_alloc_interleaved((n) * (s))
#define free_numa(p, s) numa_free((p), (s))
#else
#define malloc_numa(s) malloc(s)
#define calloc_numa(n, s) calloc((n), (s))
#define free_numa(p, s) free(p)
#endif

extern void read_fits(const char *filename, double **data, long *n_samples, long *p_dim, int *xa, int *ya, int *nax);
extern void read_fits_float(const char *filename, float **data, long *n_samples, long *p_dim, int *xa, int *ya, int *nax);
extern void write_fits_2d(const char *filename, double *data, int cols, int rows);
extern void write_fits_2d_float(const char *filename, float *data, int cols, int rows);
extern void write_fits_3d(const char *filename, double *data, int x_dim, int y_dim, int n_samples);
extern void write_fits_3d_float(const char *filename, float *data, int x_dim, int y_dim, int n_samples);

// Generate normally distributed noise
double rand_normal() {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1 > 1e-15 ? u1 : 1e-15)) * cos(2.0 * M_PI * u2);
}

void quadratic_expand_double(double *X, double *Z, long n_samples, int nx) {
    int z_dim = nx + nx * (nx + 1) / 2;
    #pragma omp parallel for
    for (long i = 0; i < n_samples; i++) {
        memcpy(&Z[i * z_dim], &X[i * nx], nx * sizeof(double));
        int idx = nx;
        for (int j = 0; j < nx; j++) {
            for (int k = j; k < nx; k++) {
                Z[i * z_dim + idx++] = X[i * nx + j] * X[i * nx + k];
            }
        }
    }
}

void quadratic_expand_float(float *X, float *Z, long n_samples, int nx) {
    int z_dim = nx + nx * (nx + 1) / 2;
    #pragma omp parallel for
    for (long i = 0; i < n_samples; i++) {
        memcpy(&Z[i * z_dim], &X[i * nx], nx * sizeof(float));
        int idx = nx;
        for (int j = 0; j < nx; j++) {
            for (int k = j; k < nx; k++) {
                Z[i * z_dim + idx++] = X[i * nx + j] * X[i * nx + k];
            }
        }
    }
}

void print_help(const char *prog) {
    printf("Usage: %s <new_X.fits> <new_Y.fits> <model_prefix> [-float]\n", prog);
    printf("  Updates the given model with new data using Recursive Least Squares.\n");
    printf("  Note: The model must have been created with -save_update in wfs2psf_v2_fit.\n");
}

int main(int argc, char **argv) {
    if (argc < 4) {
        print_help(argv[0]);
        return 1;
    }

    const char *x_file = argv[1];
    const char *y_file = argv[2];
    const char *model_prefix = argv[3];

    int use_float = 0;
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "-float") == 0) use_float = 1;
    }

    if (!use_float) {
        // ===================================
        // DOUBLE PRECISION PATH
        // ===================================
        printf("Running DOUBLE precision update...\n");
        double *X = NULL, *Y = NULL;
        long N, P_X, N_Y, P_Y;
        int xa_x, ya_x, nax_x;
        int xa_y, ya_y, nax_y;

        read_fits(x_file, &X, &N, &P_X, &xa_x, &ya_x, &nax_x);
        read_fits(y_file, &Y, &N_Y, &P_Y, &xa_y, &ya_y, &nax_y);
        
        if (N != N_Y) {
            fprintf(stderr, "Error: New X has %ld frames but new Y has %ld frames.\n", N, N_Y);
            return 1;
        }

        char path[1024];
        double *X_mean, *X_std, *Y_mean, *Y_std, *PCx, *PCy_local = NULL;
        double *ZtZ_total, *ZtU_total, *B_latent = NULL;
        long n1, p1, nx, p2;
        int xa, ya, nax;

        snprintf(path, 1024, "%s_Xmean.fits", model_prefix);
        read_fits(path, &X_mean, &n1, &p1, &xa, &ya, &nax);
        if (p1 != P_X) { fprintf(stderr, "Feature mismatch X\n"); return 1; }

        snprintf(path, 1024, "%s_Xstd.fits", model_prefix);
        read_fits(path, &X_std, &n1, &p1, &xa, &ya, &nax);

        snprintf(path, 1024, "%s_Ymean.fits", model_prefix);
        read_fits(path, &Y_mean, &n1, &p2, &xa_y, &ya_y, &nax_y);
        if (n1*p2 != P_Y) { fprintf(stderr, "Feature mismatch Y\n"); return 1; }

        snprintf(path, 1024, "%s_Ystd.fits", model_prefix);
        read_fits(path, &Y_std, &n1, &p2, &xa, &ya, &nax);

        snprintf(path, 1024, "%s_PCx.fits", model_prefix);
        read_fits(path, &PCx, &n1, &nx, &xa, &ya, &nax);

        snprintf(path, 1024, "%s_ZtZ.fits", model_prefix);
        long z_dim, z_dim2;
        read_fits(path, &ZtZ_total, &z_dim, &z_dim2, &xa, &ya, &nax);
        if (z_dim != z_dim2) { fprintf(stderr, "ZtZ matrix not loaded correctly\n"); return 1; }

        snprintf(path, 1024, "%s_ZtU.fits", model_prefix);
        long z_dim3, target_dim;
        read_fits(path, &ZtU_total, &z_dim3, &target_dim, &xa, &ya, &nax);

        // Load v2 info
        int patchsize=16, ny_per_patch=5, nxp=0, nyp=0, y_pca_mode = 1;
        int N_old = 0, scale = 1, use_quadratic = 1;
        double reg_factor = 1.0;
        snprintf(path, 1024, "%s_v2_info.txt", model_prefix);
        FILE *fp = fopen(path, "r");
        if (!fp) { fprintf(stderr, "Error: %s not found.\n", path); exit(1); }
        fscanf(fp, "patchsize %d\n", &patchsize);
        fscanf(fp, "ny_per_patch %d\n", &ny_per_patch);
        fscanf(fp, "nxp %d\n", &nxp);
        fscanf(fp, "nyp %d\n", &nyp);
        fscanf(fp, "y_pca_mode %d\n", &y_pca_mode);
        fscanf(fp, "N_total %d\n", &N_old);
        fscanf(fp, "reg_factor %lf\n", &reg_factor);
        fscanf(fp, "scale %d\n", &scale);
        fscanf(fp, "use_quadratic %d\n", &use_quadratic);
        fclose(fp);

        if (y_pca_mode) {
            snprintf(path, 1024, "%s_PCy_local.fits", model_prefix);
            long n_p_l, p_dim_l;
            read_fits(path, &PCy_local, &n_p_l, &p_dim_l, &xa, &ya, &nax);
        }

        // --- Data Transform --- //
        for (long i = 0; i < N; i++) {
            for (long j = 0; j < P_X; j++) {
                X[i * P_X + j] -= X_mean[j];
                if (scale) X[i * P_X + j] /= X_std[j];
            }
            for (long j = 0; j < P_Y; j++) {
                Y[i * P_Y + j] -= Y_mean[j];
                if (scale) Y[i * P_Y + j] /= Y_std[j];
            }
        }

        double *T = (double *)malloc_numa(N * nx * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N, (int)nx, (int)P_X, 1.0, X, (int)P_X, PCx, (int)nx, 0.0, T, (int)nx);

        double *Z = (double *)malloc_numa(N * z_dim * sizeof(double));
        if (use_quadratic) quadratic_expand_double(T, Z, N, nx);
        else memcpy(Z, T, N * nx * sizeof(double));
        
        // --- Transform Y to U_new --- //
        int n_patches = nxp * nyp;
        int patch_pixels = patchsize * patchsize;
        int target_dim_total = y_pca_mode ? n_patches * ny_per_patch : n_patches * patch_pixels;
        double *U_latent = (double *)malloc_numa(N * target_dim_total * sizeof(double));

        if (y_pca_mode) {
            #pragma omp parallel for
            for (int patch_idx = 0; patch_idx < n_patches; patch_idx++) {
                int py = patch_idx / nxp;
                int px = patch_idx % nxp;
                double *Y_patch = (double *)malloc_numa(N * patch_pixels * sizeof(double));
                for (int i = 0; i < N; i++) {
                    for (int dy = 0; dy < patchsize; dy++) {
                        for (int dx = 0; dx < patchsize; dx++) {
                            Y_patch[i * patch_pixels + dy * patchsize + dx] = 
                                Y[i * P_Y + (py * patchsize + dy) * xa_y + (px * patchsize + dx)];
                        }
                    }
                }
                double *PCy_patch = &PCy_local[patch_idx * (patch_pixels * ny_per_patch)];
                double *U_this = (double *)malloc_numa(N * ny_per_patch * sizeof(double));
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, ny_per_patch, patch_pixels,
                            1.0, Y_patch, patch_pixels, PCy_patch, ny_per_patch, 0.0, U_this, ny_per_patch);
                for (int i = 0; i < N; i++) {
                    for (int k = 0; k < ny_per_patch; k++) {
                        U_latent[i * target_dim_total + patch_idx * ny_per_patch + k] = U_this[i * ny_per_patch + k];
                    }
                }
                free_numa(Y_patch, N * patch_pixels * sizeof(double));
                free_numa(U_this, N * ny_per_patch * sizeof(double));
            }
        } else {
            #pragma omp parallel for
            for (int patch_idx = 0; patch_idx < n_patches; patch_idx++) {
                int py = patch_idx / nxp;
                int px = patch_idx % nxp;
                for (int i = 0; i < N; i++) {
                    for (int dy = 0; dy < patchsize; dy++) {
                        for (int dx = 0; dx < patchsize; dx++) {
                            int k = dy * patchsize + dx;
                            U_latent[i * target_dim_total + patch_idx * patch_pixels + k] = 
                                Y[i * P_Y + (py * patchsize + dy) * xa_y + (px * patchsize + dx)];
                        }
                    }
                }
            }
        }

        // --- Incremental Math --- //
        printf("Updating Matrix...\n");
        // Sanity check: target_dim from ZtU.fits should match computed target_dim_total
        if ((long)target_dim_total != target_dim) {
            fprintf(stderr, "Error: target_dim mismatch: ZtU.fits has %ld cols but current config gives %d. Model may be from a different fit.\n", target_dim, target_dim_total);
            return 1;
        }

        double *ZtZ_new = (double *)calloc_numa(z_dim * z_dim, sizeof(double));
        double *ZtU_new = (double *)calloc_numa(z_dim * target_dim_total, sizeof(double));

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, z_dim, z_dim, N, 1.0, Z, z_dim, Z, z_dim, 0.0, ZtZ_new, z_dim);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, z_dim, target_dim_total, N, 1.0, Z, z_dim, U_latent, target_dim_total, 0.0, ZtU_new, target_dim_total);

        for (int i = 0; i < z_dim * z_dim; i++) ZtZ_total[i] += ZtZ_new[i];
        for (int i = 0; i < z_dim * target_dim; i++) ZtU_total[i] += ZtU_new[i];

        // --- Regression Step --- //
        B_latent = (double *)malloc_numa(z_dim * target_dim * sizeof(double));
        double *ZtZ_calc = (double *)malloc_numa(z_dim * z_dim * sizeof(double));
        memcpy(ZtZ_calc, ZtZ_total, z_dim * z_dim * sizeof(double));
        
        if (reg_factor > 0.0) {
            double *ZtZ_svd = (double *)malloc_numa(z_dim * z_dim * sizeof(double));
            memcpy(ZtZ_svd, ZtZ_total, z_dim * z_dim * sizeof(double));
            double *S_ztz = (double *)malloc_numa(z_dim * sizeof(double));
            LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'N', z_dim, z_dim, ZtZ_svd, z_dim, S_ztz, NULL, 1, NULL, 1);
            double ridge_auto = reg_factor * (DBL_EPSILON * z_dim * S_ztz[0]);
            for (int i = 0; i < z_dim; i++) ZtZ_calc[i * z_dim + i] += ridge_auto;
            free_numa(ZtZ_svd, z_dim * z_dim * sizeof(double));
            free_numa(S_ztz, z_dim * sizeof(double));
        }
        
        double *ZtU_calc = (double *)malloc_numa(z_dim * target_dim * sizeof(double));
        memcpy(ZtU_calc, ZtU_total, z_dim * target_dim * sizeof(double));
        
        LAPACKE_dposv(LAPACK_ROW_MAJOR, 'U', z_dim, target_dim, ZtZ_calc, z_dim, ZtU_calc, target_dim);
        memcpy(B_latent, ZtU_calc, z_dim * target_dim * sizeof(double));

        // --- Save and overwrite old matrices --- //
        snprintf(path, 1024, "%s_B_latent.fits", model_prefix);
        write_fits_2d(path, B_latent, target_dim, z_dim); 
        snprintf(path, 1024, "%s_ZtZ.fits", model_prefix);
        write_fits_2d(path, ZtZ_total, z_dim, z_dim);
        snprintf(path, 1024, "%s_ZtU.fits", model_prefix);
        write_fits_2d(path, ZtU_total, target_dim, z_dim);

        int N_new_total = N_old + N;
        snprintf(path, 1024, "%s_v2_info.txt", model_prefix);
        fp = fopen(path, "w");
        fprintf(fp, "patchsize %d\n", patchsize);
        fprintf(fp, "ny_per_patch %d\n", ny_per_patch);
        fprintf(fp, "nxp %d\n", nxp);
        fprintf(fp, "nyp %d\n", nyp);
        fprintf(fp, "y_pca_mode %d\n", y_pca_mode);
        fprintf(fp, "N_total %d\n", N_new_total);
        fprintf(fp, "reg_factor %f\n", reg_factor);
        fprintf(fp, "scale %d\n", scale);
        fprintf(fp, "use_quadratic %d\n", use_quadratic);
        fclose(fp);

        printf("Successfully updated model %s from %d to %d frames.\n", model_prefix, N_old, N_new_total);

        free(X_mean); free(X_std); free(Y_mean); free(Y_std); free(PCx); free(ZtZ_total); free(ZtU_total);
        if (PCy_local) free(PCy_local);
        free_numa(X, N * P_X * sizeof(double)); free_numa(Y, N * P_Y * sizeof(double));
        free_numa(T, N * nx * sizeof(double)); free_numa(Z, N * z_dim * sizeof(double));
        free_numa(U_latent, N * target_dim_total * sizeof(double));
        free_numa(ZtZ_new, z_dim * z_dim * sizeof(double)); free_numa(ZtU_new, z_dim * target_dim_total * sizeof(double));
        free_numa(ZtZ_calc, z_dim * z_dim * sizeof(double)); free_numa(ZtU_calc, z_dim * target_dim * sizeof(double));
        free_numa(B_latent, z_dim * target_dim * sizeof(double));

    } else {
        // ===================================
        // SINGLE PRECISION PATH
        // ===================================
        printf("Running SINGLE precision update...\n");
        float *X = NULL, *Y = NULL;
        long N, P_X, N_Y, P_Y;
        int xa_x, ya_x, nax_x;
        int xa_y, ya_y, nax_y;

        read_fits_float(x_file, &X, &N, &P_X, &xa_x, &ya_x, &nax_x);
        read_fits_float(y_file, &Y, &N_Y, &P_Y, &xa_y, &ya_y, &nax_y);
        
        if (N != N_Y) {
            fprintf(stderr, "Error: New X has %ld frames but new Y has %ld frames.\n", N, N_Y);
            return 1;
        }

        char path[1024];
        float *X_mean, *X_std, *Y_mean, *Y_std, *PCx, *PCy_local = NULL;
        float *ZtZ_total, *ZtU_total, *B_latent = NULL;
        long n1, p1, nx, p2;
        int xa, ya, nax;

        snprintf(path, 1024, "%s_Xmean.fits", model_prefix);
        read_fits_float(path, &X_mean, &n1, &p1, &xa, &ya, &nax);
        if (p1 != P_X) { fprintf(stderr, "Feature mismatch X\n"); return 1; }

        snprintf(path, 1024, "%s_Xstd.fits", model_prefix);
        read_fits_float(path, &X_std, &n1, &p1, &xa, &ya, &nax);

        snprintf(path, 1024, "%s_Ymean.fits", model_prefix);
        read_fits_float(path, &Y_mean, &n1, &p2, &xa_y, &ya_y, &nax_y);

        snprintf(path, 1024, "%s_Ystd.fits", model_prefix);
        read_fits_float(path, &Y_std, &n1, &p2, &xa, &ya, &nax);

        snprintf(path, 1024, "%s_PCx.fits", model_prefix);
        read_fits_float(path, &PCx, &n1, &nx, &xa, &ya, &nax);

        snprintf(path, 1024, "%s_ZtZ.fits", model_prefix);
        long z_dim, z_dim2;
        read_fits_float(path, &ZtZ_total, &z_dim, &z_dim2, &xa, &ya, &nax);

        snprintf(path, 1024, "%s_ZtU.fits", model_prefix);
        long z_dim3, target_dim;
        read_fits_float(path, &ZtU_total, &z_dim3, &target_dim, &xa, &ya, &nax);

        int patchsize=16, ny_per_patch=5, nxp=0, nyp=0, y_pca_mode = 1;
        int N_old = 0, scale = 1, use_quadratic = 1;
        float reg_factor = 1.0f;
        snprintf(path, 1024, "%s_v2_info.txt", model_prefix);
        FILE *fp = fopen(path, "r");
        if (!fp) { fprintf(stderr, "Error: %s not found.\n", path); exit(1); }
        fscanf(fp, "patchsize %d\n", &patchsize);
        fscanf(fp, "ny_per_patch %d\n", &ny_per_patch);
        fscanf(fp, "nxp %d\n", &nxp);
        fscanf(fp, "nyp %d\n", &nyp);
        fscanf(fp, "y_pca_mode %d\n", &y_pca_mode);
        fscanf(fp, "N_total %d\n", &N_old);
        fscanf(fp, "reg_factor %f\n", &reg_factor);
        fscanf(fp, "scale %d\n", &scale);
        fscanf(fp, "use_quadratic %d\n", &use_quadratic);
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
            for (long j = 0; j < P_Y; j++) {
                Y[i * P_Y + j] -= Y_mean[j];
                if (scale) Y[i * P_Y + j] /= Y_std[j];
            }
        }

        float *T = (float *)malloc_numa(N * nx * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N, (int)nx, (int)P_X, 1.0f, X, (int)P_X, PCx, (int)nx, 0.0f, T, (int)nx);

        float *Z = (float *)malloc_numa(N * z_dim * sizeof(float));
        if (use_quadratic) quadratic_expand_float(T, Z, N, nx);
        else memcpy(Z, T, N * nx * sizeof(float));
        
        int n_patches = nxp * nyp;
        int patch_pixels = patchsize * patchsize;
        int target_dim_total = y_pca_mode ? n_patches * ny_per_patch : n_patches * patch_pixels;
        float *U_latent = (float *)malloc_numa(N * target_dim_total * sizeof(float));

        if (y_pca_mode) {
            #pragma omp parallel for
            for (int patch_idx = 0; patch_idx < n_patches; patch_idx++) {
                int py = patch_idx / nxp;
                int px = patch_idx % nxp;
                float *Y_patch = (float *)malloc_numa(N * patch_pixels * sizeof(float));
                for (int i = 0; i < N; i++) {
                    for (int dy = 0; dy < patchsize; dy++) {
                        for (int dx = 0; dx < patchsize; dx++) {
                            Y_patch[i * patch_pixels + dy * patchsize + dx] = 
                                Y[i * P_Y + (py * patchsize + dy) * xa_y + (px * patchsize + dx)];
                        }
                    }
                }
                float *PCy_patch = &PCy_local[patch_idx * (patch_pixels * ny_per_patch)];
                float *U_this = (float *)malloc_numa(N * ny_per_patch * sizeof(float));
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, ny_per_patch, patch_pixels,
                            1.0f, Y_patch, patch_pixels, PCy_patch, ny_per_patch, 0.0f, U_this, ny_per_patch);
                for (int i = 0; i < N; i++) {
                    for (int k = 0; k < ny_per_patch; k++) {
                        U_latent[i * target_dim_total + patch_idx * ny_per_patch + k] = U_this[i * ny_per_patch + k];
                    }
                }
                free_numa(Y_patch, N * patch_pixels * sizeof(float));
                free_numa(U_this, N * ny_per_patch * sizeof(float));
            }
        }

        printf("Updating Matrix...\n");
        // Sanity check: target_dim from ZtU.fits should match computed target_dim_total
        if ((long)target_dim_total != target_dim) {
            fprintf(stderr, "Error: target_dim mismatch: ZtU.fits has %ld cols but current config gives %d. Model may be from a different fit.\n", target_dim, target_dim_total);
            return 1;
        }
        float *ZtZ_new = (float *)calloc_numa(z_dim * z_dim, sizeof(float));
        float *ZtU_new = (float *)calloc_numa(z_dim * target_dim_total, sizeof(float));

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, z_dim, z_dim, N, 1.0f, Z, z_dim, Z, z_dim, 0.0f, ZtZ_new, z_dim);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, z_dim, target_dim_total, N, 1.0f, Z, z_dim, U_latent, target_dim_total, 0.0f, ZtU_new, target_dim_total);

        for (int i = 0; i < z_dim * z_dim; i++) ZtZ_total[i] += ZtZ_new[i];
        for (int i = 0; i < z_dim * target_dim; i++) ZtU_total[i] += ZtU_new[i];

        B_latent = (float *)malloc_numa(z_dim * target_dim * sizeof(float));
        float *ZtZ_calc = (float *)malloc_numa(z_dim * z_dim * sizeof(float));
        memcpy(ZtZ_calc, ZtZ_total, z_dim * z_dim * sizeof(float));
        
        if (reg_factor > 0.0f) {
            float *ZtZ_svd = (float *)malloc_numa(z_dim * z_dim * sizeof(float));
            memcpy(ZtZ_svd, ZtZ_total, z_dim * z_dim * sizeof(float));
            float *S_ztz = (float *)malloc_numa(z_dim * sizeof(float));
            LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'N', z_dim, z_dim, ZtZ_svd, z_dim, S_ztz, NULL, 1, NULL, 1);
            float ridge_auto = reg_factor * (FLT_EPSILON * z_dim * S_ztz[0]);
            for (int i = 0; i < z_dim; i++) ZtZ_calc[i * z_dim + i] += ridge_auto;
            free_numa(ZtZ_svd, z_dim * z_dim * sizeof(float));
            free_numa(S_ztz, z_dim * sizeof(float));
        }
        
        float *ZtU_calc = (float *)malloc_numa(z_dim * target_dim * sizeof(float));
        memcpy(ZtU_calc, ZtU_total, z_dim * target_dim * sizeof(float));
        
        LAPACKE_sposv(LAPACK_ROW_MAJOR, 'U', z_dim, target_dim, ZtZ_calc, z_dim, ZtU_calc, target_dim);
        memcpy(B_latent, ZtU_calc, z_dim * target_dim * sizeof(float));

        snprintf(path, 1024, "%s_B_latent.fits", model_prefix);
        write_fits_2d_float(path, B_latent, target_dim, z_dim); 
        snprintf(path, 1024, "%s_ZtZ.fits", model_prefix);
        write_fits_2d_float(path, ZtZ_total, z_dim, z_dim);
        snprintf(path, 1024, "%s_ZtU.fits", model_prefix);
        write_fits_2d_float(path, ZtU_total, target_dim, z_dim);

        int N_new_total = N_old + N;
        snprintf(path, 1024, "%s_v2_info.txt", model_prefix);
        fp = fopen(path, "w");
        fprintf(fp, "patchsize %d\n", patchsize);
        fprintf(fp, "ny_per_patch %d\n", ny_per_patch);
        fprintf(fp, "nxp %d\n", nxp);
        fprintf(fp, "nyp %d\n", nyp);
        fprintf(fp, "y_pca_mode %d\n", y_pca_mode);
        fprintf(fp, "N_total %d\n", N_new_total);
        fprintf(fp, "reg_factor %f\n", reg_factor);
        fprintf(fp, "scale %d\n", scale);
        fprintf(fp, "use_quadratic %d\n", use_quadratic);
        fclose(fp);

        printf("Successfully updated model %s from %d to %d frames.\n", model_prefix, N_old, N_new_total);

        free(X_mean); free(X_std); free(Y_mean); free(Y_std); free(PCx); free(ZtZ_total); free(ZtU_total);
        if (PCy_local) free(PCy_local);
        free_numa(X, N * P_X * sizeof(float)); free_numa(Y, N * P_Y * sizeof(float));
        free_numa(T, N * nx * sizeof(float)); free_numa(Z, N * z_dim * sizeof(float));
        free_numa(U_latent, N * target_dim_total * sizeof(float));
        free_numa(ZtZ_new, z_dim * z_dim * sizeof(float)); free_numa(ZtU_new, z_dim * target_dim_total * sizeof(float));
        free_numa(ZtZ_calc, z_dim * z_dim * sizeof(float)); free_numa(ZtU_calc, z_dim * target_dim * sizeof(float));
        free_numa(B_latent, z_dim * target_dim * sizeof(float));
    }

    return 0;
}
