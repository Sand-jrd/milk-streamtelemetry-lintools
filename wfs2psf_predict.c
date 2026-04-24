#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <unistd.h>
#include "common.h"

#ifdef USE_NUMA
#include <numa.h>
#define malloc_numa(s) numa_alloc_interleaved(s)
#define calloc_numa(n, s) ({ \
    size_t _size = (n) * (s); \
    void *_ptr = numa_alloc_interleaved(_size); \
    if (_ptr) memset(_ptr, 0, _size); \
    _ptr; \
})
#define free_numa(p, s) numa_free(p, s)
#else
#define malloc_numa(s) malloc(s)
#define calloc_numa(n, s) calloc(n, s)
#define free_numa(p, s) free(p)
#endif

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
        int patchsize, ny_per_patch, nxp, nyp, y_pca_mode = 1, normalized = 0;
        snprintf(path, 1024, "%s_v2_info.txt", model_prefix);
        FILE *fp = fopen(path, "r");
        if (!fp) { fprintf(stderr, "Error: %s not found. Model might not be v2.\n", path); exit(1); }
        char line[256];
        while(fgets(line, sizeof(line), fp)) {
            if(sscanf(line, "patchsize %d", &patchsize) == 1);
            else if(sscanf(line, "ny_per_patch %d", &ny_per_patch) == 1);
            else if(sscanf(line, "nxp %d", &nxp) == 1);
            else if(sscanf(line, "nyp %d", &nyp) == 1);
            else if(sscanf(line, "y_pca_mode %d", &y_pca_mode) == 1);
            else if(sscanf(line, "normalized %d", &normalized) == 1);
        }
        fclose(fp);

        if (normalized) printf("INFO: This model was trained on NORMALIZED frames (unit sum).\n");

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

        double *T = (double *)malloc_numa(N * nx * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N, (int)nx, (int)P_X, 1.0, X, (int)P_X, PCx, (int)nx, 0.0, T, (int)nx);

        // === WFS Coverage Diagnostic ===
        // Load training singular values (saved by fit as PCx_sv.fits)
        snprintf(path, 1024, "%s_PCx_sv.fits", model_prefix);
        {
            FILE *sv_check = fopen(path, "r");
            long n_sv_1, n_sv_2; int xa_sv, ya_sv, nax_sv;
            if (sv_check) {
                fclose(sv_check);
                double *S_train = NULL;
                read_fits(path, &S_train, &n_sv_1, &n_sv_2, &xa_sv, &ya_sv, &nax_sv);
                int N_train_est = 0;
                // Estimate N_train from info file: read N_total field
                char info_path[1024];
                snprintf(info_path, 1024, "%s_v2_info.txt", model_prefix);
                FILE *fp2 = fopen(info_path, "r");
                char key[64]; int val_i; float val_f;
                if (fp2) {
                    char line[256];
                    while (fgets(line, sizeof(line), fp2)) {
                        if (sscanf(line, "N_total %d", &val_i) == 1) N_train_est = val_i;
                    }
                    fclose(fp2);
                }
                if (N_train_est < 2) N_train_est = 1000; // fallback
                double sqrt_N = sqrt((double)N_train_est - 1.0);

                printf("\n=== WFS COVERAGE DIAGNOSTIC =======================\n");
                printf("%-6s  %-12s  %-12s  %-8s  %s\n",
                       "Mode", "Train_std", "Sci_std", "Ratio", "Status");
                int n_extrap = 0;
                for (long k = 0; k < nx; k++) {
                    double train_std = S_train[k] / sqrt_N;
                    // Compute science score std for mode k
                    double sci_mean = 0;
                    for (long i = 0; i < N; i++) sci_mean += T[i*nx + k];
                    sci_mean /= N;
                    double sci_std = 0;
                    for (long i = 0; i < N; i++) { double d = T[i*nx+k]-sci_mean; sci_std += d*d; }
                    sci_std = sqrt(sci_std / (N > 1 ? N-1 : 1));
                    double ratio = (train_std > 1e-12) ? sci_std / train_std : 0.0;
                    const char *status = (ratio > 3.0) ? "** EXTRAPOLATION **" :
                                         (ratio > 1.5) ? "* caution *" : "OK";
                    if (ratio > 3.0) n_extrap++;
                    if (k < 20 || ratio > 1.5)  // print top 20 + any problematic modes
                        printf("PC%-4ld  %-12.4f  %-12.4f  %-8.2f  %s\n",
                               k, train_std, sci_std, ratio, status);
                }
                printf("  Modes in extrapolation (ratio>3): %d / %ld\n", n_extrap, nx);
                if (n_extrap == 0)
                    printf("  OK: Science WFS is within training distribution.\n");
                else
                    printf("  WARNING: %d WFS modes outside training range — predictions in those modes may be unreliable.\n", n_extrap);
                printf("===================================================\n\n");
                free(S_train);
            } else {
                printf("(WFS coverage check skipped: %s not found. Re-run fit to generate it.)\n", path);
            }
        }

        double *Z = T;
        if (use_quadratic) {
            Z = (double *)malloc_numa(N * z_dim * sizeof(double));
            quadratic_expand_double(T, Z, N, (int)nx);
        }

        int n_patches = nxp * nyp;
        int *pcy_offset = (int *)malloc_numa(n_patches * sizeof(int));
        int *u_offset   = (int *)malloc_numa(n_patches * sizeof(int));
        long target_dim_total = 0;
        long current_pcy_offset = 0;

        for (int py = 0; py < nyp; py++) {
            for (int px = 0; px < nxp; px++) {
                int patch_idx = py * nxp + px;
                int pw = (px == nxp - 1) ? xa_y - px * patchsize : patchsize;
                int ph = (py == nyp - 1) ? ya_y - py * patchsize : patchsize;
                int p_pixels = pw * ph;
                
                pcy_offset[patch_idx] = current_pcy_offset;
                u_offset[patch_idx] = target_dim_total;

                if (y_pca_mode) {
                    target_dim_total += ny_per_patch; 
                    current_pcy_offset += p_pixels * ny_per_patch;
                } else {
                    target_dim_total += p_pixels; 
                }
            }
        }

        double *U_pred = (double *)malloc_numa(N * target_dim * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N, (int)target_dim, (int)z_dim, 1.0, Z, (int)z_dim, B_latent, (int)target_dim, 0.0, U_pred, (int)target_dim);

        double *Y_new = (double *)calloc_numa(N * P_Y, sizeof(double));
        for (int i = 0; i < N; i++) {
            for (int py = 0; py < nyp; py++) {
                for (int px = 0; px < nxp; px++) {
                    int patch_idx = py * nxp + px;
                    int pw = (px == nxp - 1) ? xa_y - px * patchsize : patchsize;
                    int ph = (py == nyp - 1) ? ya_y - py * patchsize : patchsize;
                    int p_pixels = pw * ph;
                    
                    double *U_patch = &U_pred[i * target_dim + u_offset[patch_idx]];
                    if (y_pca_mode) {
                        double *PCy_patch = &PCy_local[pcy_offset[patch_idx]];
                        for (int dy = 0; dy < ph; dy++) {
                            for (int dx = 0; dx < pw; dx++) {
                                double val = 0;
                                int pixel_idx = dy * pw + dx;
                                for (int k = 0; k < ny_per_patch; k++) {
                                    val += U_patch[k] * PCy_patch[pixel_idx * ny_per_patch + k];
                                }
                                Y_new[i * P_Y + (py * patchsize + dy) * xa_y + (px * patchsize + dx)] = val;
                            }
                        }
                    } else {
                        // Raw pixels
                        for (int k = 0; k < p_pixels; k++) {
                            int dy = k / pw;
                            int dx = k % pw;
                            Y_new[i * P_Y + (py * patchsize + dy) * xa_y + (px * patchsize + dx)] = U_patch[k];
                        }
                    }
                }
            }
        }

        // Denormalize the fully covered image
        {
            for (long i = 0; i < N; i++) {
                for (long j = 0; j < P_Y; j++) {
                    if (y_pca_mode) Y_new[i * P_Y + j] = Y_new[i * P_Y + j] * Y_std[j] + Y_mean[j];
                    else Y_new[i * P_Y + j] += Y_mean[j];
                }
            }
        }

        if (nax_y == 3) write_fits_3d(out_file, Y_new, xa_y, ya_y, (int)N);
        else write_fits_2d(out_file, Y_new, (int)P_Y, (int)N);

        free(X); free(X_mean); free(X_std); free(Y_mean); free(Y_std);
        free(PCx); free(B_latent); if (y_pca_mode) free(PCy_local); 
        free_numa(T, N * nx * sizeof(double)); 
        if (use_quadratic) free_numa(Z, N * z_dim * sizeof(double));
        free_numa(Y_new, N * P_Y * sizeof(double)); 
        free_numa(U_pred, N * target_dim * sizeof(double));
        free_numa(pcy_offset, n_patches * sizeof(int));
        free_numa(u_offset, n_patches * sizeof(int));
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
        int patchsize, ny_per_patch, nxp, nyp, y_pca_mode = 1, normalized = 0;
        snprintf(path, 1024, "%s_v2_info.txt", model_prefix);
        FILE *fp = fopen(path, "r");
        if (!fp) { fprintf(stderr, "Error: %s not found. Model might not be v2.\n", path); exit(1); }
        char line[256];
        while(fgets(line, sizeof(line), fp)) {
            if(sscanf(line, "patchsize %d", &patchsize) == 1);
            else if(sscanf(line, "ny_per_patch %d", &ny_per_patch) == 1);
            else if(sscanf(line, "nxp %d", &nxp) == 1);
            else if(sscanf(line, "nyp %d", &nyp) == 1);
            else if(sscanf(line, "y_pca_mode %d", &y_pca_mode) == 1);
            else if(sscanf(line, "normalized %d", &normalized) == 1);
        }
        fclose(fp);

        if (normalized) printf("INFO: This model was trained on NORMALIZED frames (unit sum).\n");

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

        float *T = (float *)malloc_numa(N * nx * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N, (int)nx, (int)P_X, 1.0f, X, (int)P_X, PCx, (int)nx, 0.0f, T, (int)nx);

        float *Z = T;
        if (use_quadratic) {
            Z = (float *)malloc_numa(N * z_dim * sizeof(float));
            quadratic_expand_float(T, Z, N, (int)nx);
        }

        int n_patches = nxp * nyp;
        int *pcy_offset = (int *)malloc_numa(n_patches * sizeof(int));
        int *u_offset   = (int *)malloc_numa(n_patches * sizeof(int));
        long target_dim_total = 0;
        long current_pcy_offset = 0;

        for (int py = 0; py < nyp; py++) {
            for (int px = 0; px < nxp; px++) {
                int patch_idx = py * nxp + px;
                int pw = (px == nxp - 1) ? xa_y - px * patchsize : patchsize;
                int ph = (py == nyp - 1) ? ya_y - py * patchsize : patchsize;
                int p_pixels = pw * ph;
                
                pcy_offset[patch_idx] = current_pcy_offset;
                u_offset[patch_idx] = target_dim_total;

                if (y_pca_mode) {
                    target_dim_total += ny_per_patch; 
                    current_pcy_offset += p_pixels * ny_per_patch;
                } else {
                    target_dim_total += p_pixels; 
                }
            }
        }

        float *U_pred = (float *)malloc_numa(N * target_dim * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N, (int)target_dim, (int)z_dim, 1.0f, Z, (int)z_dim, B_latent, (int)target_dim, 0.0f, U_pred, (int)target_dim);

        float *Y_new = (float *)calloc_numa(N * P_Y, sizeof(float));
        for (int i = 0; i < N; i++) {
            for (int py = 0; py < nyp; py++) {
                for (int px = 0; px < nxp; px++) {
                    int patch_idx = py * nxp + px;
                    int pw = (px == nxp - 1) ? xa_y - px * patchsize : patchsize;
                    int ph = (py == nyp - 1) ? ya_y - py * patchsize : patchsize;
                    int p_pixels = pw * ph;
                    
                    float *U_patch = &U_pred[i * target_dim + u_offset[patch_idx]];
                    if (y_pca_mode) {
                        float *PCy_patch = &PCy_local[pcy_offset[patch_idx]];
                        for (int dy = 0; dy < ph; dy++) {
                            for (int dx = 0; dx < pw; dx++) {
                                float val = 0;
                                int pixel_idx = dy * pw + dx;
                                for (int k = 0; k < ny_per_patch; k++) {
                                    val += U_patch[k] * PCy_patch[pixel_idx * ny_per_patch + k];
                                }
                                Y_new[i * P_Y + (py * patchsize + dy) * xa_y + (px * patchsize + dx)] = val;
                            }
                        }
                    } else {
                        for (int k = 0; k < p_pixels; k++) {
                            int dy = k / pw;
                            int dx = k % pw;
                            Y_new[i * P_Y + (py * patchsize + dy) * xa_y + (px * patchsize + dx)] = U_patch[k];
                        }
                    }
                }
            }
        }

        // Denormalize the fully covered image
        {
            for (long i = 0; i < N; i++) {
                for (long j = 0; j < P_Y; j++) {
                    if (y_pca_mode) Y_new[i * P_Y + j] = Y_new[i * P_Y + j] * Y_std[j] + Y_mean[j];
                    else Y_new[i * P_Y + j] += Y_mean[j];
                }
            }
        }

        if (nax_y == 3) write_fits_3d_float(out_file, Y_new, xa_y, ya_y, (int)N);
        else write_fits_2d_float(out_file, Y_new, (int)P_Y, (int)N);

        free(X); free(X_mean); free(X_std); free(Y_mean); free(Y_std);
        free(PCx); free(B_latent); if (y_pca_mode) free(PCy_local); 
        free_numa(T, N * nx * sizeof(float)); 
        if (use_quadratic) free_numa(Z, N * z_dim * sizeof(float));
        free_numa(Y_new, N * P_Y * sizeof(float)); 
        free_numa(U_pred, N * target_dim * sizeof(float));
        free_numa(pcy_offset, n_patches * sizeof(int));
        free_numa(u_offset, n_patches * sizeof(int));
    }

    printf("Done. Saved %s\n", out_file);
    return 0;
}
