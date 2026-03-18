#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <float.h>
#include "common.h"

// Helper to expand T to Z
// For a vector T (size nx), the quadratic expansion Z contains:
// T_0, T_1, ..., T_{nx-1}, T_0*T_0, T_0*T_1, ..., T_{nx-1}*T_{nx-1}
// Total size: nx + nx*(nx+1)/2
void quadratic_expand_double(const double *T, double *Z, long n_samples, int nx) {
    int z_dim = nx + nx*(nx+1)/2;
    for (long i = 0; i < n_samples; i++) {
        const double *t = &T[i * nx];
        double *z = &Z[i * z_dim];
        int idx = 0;
        for (int j = 0; j < nx; j++) {
            z[idx++] = t[j];
        }
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
        for (int j = 0; j < nx; j++) {
            z[idx++] = t[j];
        }
        for (int j = 0; j < nx; j++) {
            for (int k = j; k < nx; k++) {
                z[idx++] = t[j] * t[k];
            }
        }
    }
}

// Helper: 2D Laplacian smoothing
void apply_laplacian_double(double *img, double *out, int xa, int ya) {
    for (int y = 0; y < ya; y++) {
        for (int x = 0; x < xa; x++) {
            double center = img[y * xa + x];
            double top = (y > 0) ? img[(y - 1) * xa + x] : center;
            double bottom = (y < ya - 1) ? img[(y + 1) * xa + x] : center;
            double left = (x > 0) ? img[y * xa + (x - 1)] : center;
            double right = (x < xa - 1) ? img[y * xa + (x + 1)] : center;
            out[y * xa + x] = top + bottom + left + right - 4.0 * center;
        }
    }
}

void apply_laplacian_float(float *img, float *out, int xa, int ya) {
    for (int y = 0; y < ya; y++) {
        for (int x = 0; x < xa; x++) {
            float center = img[y * xa + x];
            float top = (y > 0) ? img[(y - 1) * xa + x] : center;
            float bottom = (y < ya - 1) ? img[(y + 1) * xa + x] : center;
            float left = (x > 0) ? img[y * xa + (x - 1)] : center;
            float right = (x < xa - 1) ? img[y * xa + (x + 1)] : center;
            out[y * xa + x] = top + bottom + left + right - 4.0f * center;
        }
    }
}

// Helpers for pseudo-inverse using SVD
// Computes A^+ (n x m) for A (m x n), row-major.
// Actually we only need pinv(Z) * U_latent which can be done via SVD + DGELSD
void solve_least_squares_double(double *Z, double *U, double *B, int n_samples, int z_dim, int target_dim) {
    // LAPACKE_dgelsd for min || Z B - U ||_2
    // Z is (n_samples x z_dim)
    // U is (n_samples x target_dim)
    // On entry, Z and U are overridden, so we need copies.
    double *Z_copy = (double *)malloc(n_samples * z_dim * sizeof(double));
    memcpy(Z_copy, Z, n_samples * z_dim * sizeof(double));
    
    // dgelsd requires B to be at least max(n_samples, z_dim) rows.
    int ldb_rows = (n_samples > z_dim) ? n_samples : z_dim;
    double *U_copy = (double *)calloc(ldb_rows * target_dim, sizeof(double));
    for(int i=0; i<n_samples; i++) {
        for(int j=0; j<target_dim; j++) {
            U_copy[i * target_dim + j] = U[i * target_dim + j];
        }
    }

    double *S = (double *)malloc(z_dim * sizeof(double));
    double rcond = -1.0; // default machine precision
    int rank;

    LAPACKE_dgelsd(LAPACK_ROW_MAJOR, n_samples, z_dim, target_dim, Z_copy, z_dim, U_copy, target_dim, S, rcond, &rank);

    // The first z_dim rows of U_copy contain the solution B
    for(int i=0; i<z_dim; i++) {
        for(int j=0; j<target_dim; j++) {
            B[i * target_dim + j] = U_copy[i * target_dim + j];
        }
    }

    free(Z_copy); free(U_copy); free(S);
}

void solve_least_squares_float(float *Z, float *U, float *B, int n_samples, int z_dim, int target_dim) {
    float *Z_copy = (float *)malloc(n_samples * z_dim * sizeof(float));
    memcpy(Z_copy, Z, n_samples * z_dim * sizeof(float));
    
    int ldb_rows = (n_samples > z_dim) ? n_samples : z_dim;
    float *U_copy = (float *)calloc(ldb_rows * target_dim, sizeof(float));
    for(int i=0; i<n_samples; i++) {
        for(int j=0; j<target_dim; j++) {
            U_copy[i * target_dim + j] = U[i * target_dim + j];
        }
    }

    float *S = (float *)malloc(z_dim * sizeof(float));
    float rcond = -1.0f;
    int rank;

    LAPACKE_sgelsd(LAPACK_ROW_MAJOR, n_samples, z_dim, target_dim, Z_copy, z_dim, U_copy, target_dim, S, rcond, &rank);

    for(int i=0; i<z_dim; i++) {
        for(int j=0; j<target_dim; j++) {
            B[i * target_dim + j] = U_copy[i * target_dim + j];
        }
    }

    free(Z_copy); free(U_copy); free(S);
}

// Generate normally distributed noise (Box-Muller)
double rand_normal() {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1 > 1e-15 ? u1 : 1e-15)) * cos(2.0 * M_PI * u2);
}

void print_help(const char *prog) {
    printf("Usage: %s <X_file.fits> <Y_file.fits> <out_prefix> [options]\n", prog);
    printf("Options:\n");
    printf("  -nx <int>             n_components_x (default: 10, -1 for auto)\n");
    printf("  -ny <int>             n_components_y per patch (default: 5, 0 for raw pixels)\n");
    printf("  -patchsize <int>      PSF patch size (default: 16)\n");
    printf("  -noscale              Disable scaling (default: enabled)\n");
    printf("  -noquad               Disable quadratic expansion (default: enabled)\n");
    printf("  -noreg                Disable ridge regularization (default: enabled)\n");
    printf("  -noise <float>        Noise std (default: 1e-9)\n");
    printf("  -laplacian <float>    Laplacian lambda (default: -1.0)\n");
    printf("  -trainsize <int>      Training set size (default: all)\n");
    printf("  -float                Use single precision (default is double)\n");
}

int main(int argc, char **argv) {
    if (argc < 4) {
        print_help(argv[0]);
        return 1;
    }

    const char *x_file = argv[1];
    const char *y_file = argv[2];
    const char *out_prefix = argv[3];

    int nx = 10;
    int ny = 5; // Default components per patch
    int patchsize = 16;
    int scale = 1;
    int use_quadratic = 1;
    int reg_mode = 1;
    double noise_std = 1e-9;
    double laplacian_lambda = -1.0;
    int train_size = -1;
    int use_float = 0;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "-nx") == 0) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-ny") == 0) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-patchsize") == 0) patchsize = atoi(argv[++i]);
        else if (strcmp(argv[i], "-noscale") == 0) scale = 0;
        else if (strcmp(argv[i], "-noquad") == 0) use_quadratic = 0;
        else if (strcmp(argv[i], "-noreg") == 0) reg_mode = 0;
        else if (strcmp(argv[i], "-noise") == 0) noise_std = atof(argv[++i]);
        else if (strcmp(argv[i], "-laplacian") == 0) laplacian_lambda = atof(argv[++i]);
        else if (strcmp(argv[i], "-trainsize") == 0) train_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "-float") == 0) use_float = 1;
    }

    printf("Settings: nx=%d, ny=%d, scale=%d, quad=%d, reg=%d, noise=%g, laplacian=%g, train_size=%d, float=%d\n",
           nx, ny, scale, use_quadratic, reg_mode, noise_std, laplacian_lambda, train_size, use_float);

    if (!use_float) {
        double *X = NULL, *Y = NULL;
        long N_X, P_X, N_Y, P_Y;
        int xa_x, ya_x, xa_y, ya_y, nax_x, nax_y;

        read_fits(x_file, &X, &N_X, &P_X, &xa_x, &ya_x, &nax_x);
        read_fits(y_file, &Y, &N_Y, &P_Y, &xa_y, &ya_y, &nax_y);

        int N = (train_size > 0 && train_size < N_X) ? train_size : N_X;
        if (N > N_Y) N = N_Y;
        
        printf("Using N = %d samples. P_X = %ld, P_Y = %ld\n", N, P_X, P_Y);

        double *X_mean = (double*)calloc(P_X, sizeof(double));
        double *Y_mean = (double*)calloc(P_Y, sizeof(double));
        double *X_std = (double*)malloc(P_X * sizeof(double));
        double *Y_std = (double*)malloc(P_Y * sizeof(double));

        for(long j=0; j<P_X; j++) X_std[j] = 1.0;
        for(long j=0; j<P_Y; j++) Y_std[j] = 1.0;

        for (int i = 0; i < N; i++) {
            for (long j = 0; j < P_X; j++) X_mean[j] += X[i * P_X + j];
            for (long j = 0; j < P_Y; j++) Y_mean[j] += Y[i * P_Y + j];
        }
        for (long j = 0; j < P_X; j++) X_mean[j] /= N;
        for (long j = 0; j < P_Y; j++) Y_mean[j] /= N;

        for (int i = 0; i < N; i++) {
            for (long j = 0; j < P_X; j++) X[i * P_X + j] -= X_mean[j];
            for (long j = 0; j < P_Y; j++) Y[i * P_Y + j] -= Y_mean[j];
        }

        if (scale) {
            for (long i = 0; i < P_X; i++) X_std[i] = 0;
            for (long i = 0; i < P_Y; i++) Y_std[i] = 0;

            for (int i = 0; i < N; i++) {
                for (long j = 0; j < P_X; j++) {
                    double v = X[i * P_X + j];
                    X_std[j] += v * v;
                }
                for (long j = 0; j < P_Y; j++) {
                    double v = Y[i * P_Y + j];
                    Y_std[j] += v * v;
                }
            }
            for (long j = 0; j < P_X; j++) {
                X_std[j] = sqrt(X_std[j] / (N - 1));
                if (X_std[j] < 1e-12) X_std[j] = 1e-12;
            }
            for (long j = 0; j < P_Y; j++) {
                Y_std[j] = sqrt(Y_std[j] / (N - 1));
                if (Y_std[j] < 1e-12) Y_std[j] = 1e-12;
            }

            for (int i = 0; i < N; i++) {
                for (long j = 0; j < P_X; j++) X[i * P_X + j] /= X_std[j];
                for (long j = 0; j < P_Y; j++) Y[i * P_Y + j] /= Y_std[j];
            }
        }

        printf("PCA on X...\n");
        // PCA on X
        // LAPACKE_dgesdd overwrites X, so let's copy it
        double *Xc_copy = (double *)malloc(N * P_X * sizeof(double));
        memcpy(Xc_copy, X, N * P_X * sizeof(double));

        int min_dim_X = N < P_X ? N : P_X;
        double *S_x = (double *)malloc(min_dim_X * sizeof(double));
        double *U_x = (double *)malloc(N * min_dim_X * sizeof(double));
        double *Vt_x = (double *)malloc(min_dim_X * P_X * sizeof(double));
        LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', N, P_X, Xc_copy, P_X, S_x, U_x, min_dim_X, Vt_x, P_X);

        if (nx == -1) {
            // Auto nx
            double sum_var = 0;
            for(int k=0; k<min_dim_X; k++) sum_var += (S_x[k]*S_x[k]) / (N-1);
            double cum_var = 0;
            nx = min_dim_X;
            for(int k=0; k<min_dim_X; k++) {
                cum_var += (S_x[k]*S_x[k]) / (N-1);
                if (cum_var / sum_var >= 0.95) {
                    nx = k + 1;
                    break;
                }
            }
            printf("WFS PC number set to %d\n", nx);
        }
        if (nx > min_dim_X) nx = min_dim_X;
        printf("PCA on X completed.\n");

        // PCx is Vt_x[:nx, :].T  -> P_X x nx
        double *PCx = (double *)malloc(P_X * nx * sizeof(double));
        for (long p = 0; p < P_X; p++) {
            for (int k = 0; k < nx; k++) {
                PCx[p * nx + k] = Vt_x[k * P_X + p];
            }
        }

        // T = X @ PCx
        double *T = (double *)malloc(N * nx * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, nx, P_X, 1.0, X, P_X, PCx, nx, 0.0, T, nx);

        double *PCy_global = NULL; // Combined basis for all patches
        double *U_latent = NULL;
        int target_dim_total = 0;
        int nxp = xa_y / patchsize;
        int nyp = ya_y / patchsize;
        int n_patches = nxp * nyp;
        int patch_pixels = patchsize * patchsize;
        int ny_per_patch = (ny <= 0) ? patch_pixels : ny;
        int y_pca_mode = (ny > 0);

        printf("Local processing on PSF: %d patches (%dx%d each), ny=%d per patch (%s)\n", 
               n_patches, patchsize, patchsize, ny_per_patch, y_pca_mode ? "PCA" : "Raw Pixels");
        
        target_dim_total = n_patches * ny_per_patch;
        U_latent = (double *)malloc(N * target_dim_total * sizeof(double));
        if (y_pca_mode) PCy_global = (double *)malloc(n_patches * patch_pixels * ny_per_patch * sizeof(double));

        for (int py = 0; py < nyp; py++) {
            for (int px = 0; px < nxp; px++) {
                int patch_idx = py * nxp + px;
                // Extract patch stack N x patch_pixels
                double *Y_patch = (double *)malloc(N * patch_pixels * sizeof(double));
                for (int i = 0; i < N; i++) {
                    for (int dy = 0; dy < patchsize; dy++) {
                        for (int dx = 0; dx < patchsize; dx++) {
                            Y_patch[i * patch_pixels + dy * patchsize + dx] = Y[i * P_Y + (py * patchsize + dy) * xa_y + (px * patchsize + dx)];
                        }
                    }
                }

                if (y_pca_mode) {
                    // PCA on Y_patch
                    double *Yc_patch = (double *)malloc(N * patch_pixels * sizeof(double));
                    memcpy(Yc_patch, Y_patch, N * patch_pixels * sizeof(double));
                    int min_dim_patch = N < patch_pixels ? N : patch_pixels;
                    double *S_p = (double *)malloc(min_dim_patch * sizeof(double));
                    double *U_p = (double *)malloc(N * min_dim_patch * sizeof(double));
                    double *Vt_p = (double *)malloc(min_dim_patch * patch_pixels * sizeof(double));
                    LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', N, patch_pixels, Yc_patch, patch_pixels, S_p, U_p, min_dim_patch, Vt_p, patch_pixels);

                    int ny_this = ny_per_patch > min_dim_patch ? min_dim_patch : ny_per_patch;
                    
                    // Store basis in PCy_global at offset
                    for (int p = 0; p < patch_pixels; p++) {
                        for (int k = 0; k < ny_this; k++) {
                            PCy_global[patch_idx * (patch_pixels * ny_per_patch) + p * ny_per_patch + k] = Vt_p[k * patch_pixels + p];
                        }
                    }

                    // Compute coefficients U_patch = Y_patch @ PCy_patch
                    double *PCy_this = &PCy_global[patch_idx * (patch_pixels * ny_per_patch)];
                    double *U_this = (double *)malloc(N * ny_per_patch * sizeof(double));
                    memset(U_this, 0, N * ny_per_patch * sizeof(double));

                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, ny_this, patch_pixels, 1.0, Y_patch, patch_pixels, PCy_this, ny_per_patch, 0.0, U_this, ny_per_patch);

                    // Copy to U_latent
                    for (int i = 0; i < N; i++) {
                        for (int k = 0; k < ny_per_patch; k++) {
                            U_latent[i * target_dim_total + patch_idx * ny_per_patch + k] = U_this[i * ny_per_patch + k];
                        }
                    }
                    free(Yc_patch); free(S_p); free(U_p); free(Vt_p); free(U_this);
                } else {
                    // No PCA, use raw pixels
                    for (int i = 0; i < N; i++) {
                        for (int k = 0; k < patch_pixels; k++) {
                            U_latent[i * target_dim_total + patch_idx * patch_pixels + k] = Y_patch[i * patch_pixels + k];
                        }
                    }
                }

                free(Y_patch);
            }
        }
        int target_dim = target_dim_total;
        printf("Local PCA on Y completed. Latent dim: %d\n", target_dim);

        int z_dim = use_quadratic ? (nx + nx*(nx+1)/2) : nx;
        double *Z = (double *)malloc(N * z_dim * sizeof(double));
        if (!Z) { fprintf(stderr, "Failed to allocate Z (N=%d, z_dim=%d)\n", (int)N, z_dim); exit(1); }
        printf("Starting quadratic expansion (z_dim=%d)...\n", z_dim);
        if (use_quadratic) {
            quadratic_expand_double(T, Z, N, nx);
        } else {
            memcpy(Z, T, N * nx * sizeof(double));
        }
        printf("Quadratic expansion completed.\n");

        int y_pca_mode = (ny > 0);
        if (!y_pca_mode && noise_std > 0) {
            for (int i = 0; i < N * z_dim; i++) {
                Z[i] += rand_normal() * noise_std;
            }
        }

        printf("Starting regression...\n");
        double *B_latent = (double *)malloc(z_dim * target_dim * sizeof(double));
        if (!B_latent) { fprintf(stderr, "Failed to allocate B_latent (%d x %d)\n", z_dim, target_dim); exit(1); }
        
        if (reg_mode) {
            double eps = DBL_EPSILON;
            if (N < z_dim) {
                // Dual ridge regression: B = Z^T (Z Z^T + lambda I)^-1 U
                double *ZZt = (double *)malloc(N * N * sizeof(double));
                if (!ZZt) { fprintf(stderr, "Failed to allocate ZZt (%d x %d)\n", (int)N, (int)N); exit(1); }
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            N, N, z_dim, 1.0, Z, z_dim, Z, z_dim, 0.0, ZZt, N);
                
                double *ZZt_copy = (double *)malloc(N * N * sizeof(double));
                if (!ZZt_copy) { fprintf(stderr, "Failed to allocate ZZt_copy\n"); exit(1); }
                memcpy(ZZt_copy, ZZt, N * N * sizeof(double));
                double *S_zzt = (double *)malloc(N * sizeof(double));
                LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'N', N, N, ZZt_copy, N, S_zzt, NULL, 1, NULL, 1);
                
                double norm_ZZt = S_zzt[0];
                double ridge_auto = eps * z_dim * norm_ZZt;
                
                for (int i = 0; i < N; i++) ZZt[i * N + i] += ridge_auto;
                
                double *M = (double *)malloc(N * target_dim * sizeof(double));
                if (!M) { fprintf(stderr, "Failed to allocate M\n"); exit(1); }
                memcpy(M, U_latent, N * target_dim * sizeof(double));
                
                // M = (Z Z^T + lambda I)^-1 U_latent
                LAPACKE_dposv(LAPACK_ROW_MAJOR, 'U', N, target_dim, ZZt, N, M, target_dim);
                
                // B_latent = Z^T M
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            z_dim, target_dim, N, 1.0, Z, z_dim, M, target_dim, 0.0, B_latent, target_dim);
                
                free(ZZt); free(ZZt_copy); free(S_zzt); free(M);
            } else {
                // Primal ridge regression: B = (Z^T Z + lambda I)^-1 Z^T U
                double *ZtZ = (double *)malloc(z_dim * z_dim * sizeof(double));
                if (!ZtZ) { fprintf(stderr, "Failed to allocate ZtZ (%d x %d)\n", z_dim, z_dim); exit(1); }
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            z_dim, z_dim, N, 1.0, Z, z_dim, Z, z_dim, 0.0, ZtZ, z_dim);
                
                double *ZtZ_copy = (double *)malloc(z_dim * z_dim * sizeof(double));
                if (!ZtZ_copy) { fprintf(stderr, "Failed to allocate ZtZ_copy\n"); exit(1); }
                memcpy(ZtZ_copy, ZtZ, z_dim * z_dim * sizeof(double));
                double *S_ztz = (double *)malloc(z_dim * sizeof(double));
                LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'N', z_dim, z_dim, ZtZ_copy, z_dim, S_ztz, NULL, 1, NULL, 1);
                
                double norm_ZtZ = S_ztz[0];
                double ridge_auto = eps * z_dim * norm_ZtZ;
    
                for (int i = 0; i < z_dim; i++) ZtZ[i * z_dim + i] += ridge_auto;
    
                double *ZtU = (double *)malloc(z_dim * target_dim * sizeof(double));
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            z_dim, target_dim, N, 1.0, Z, z_dim, U_latent, target_dim, 0.0, ZtU, target_dim);
                
                LAPACKE_dposv(LAPACK_ROW_MAJOR, 'U', z_dim, target_dim, ZtZ, z_dim, ZtU, target_dim);
                memcpy(B_latent, ZtU, z_dim * target_dim * sizeof(double));
    
                free(ZtZ); free(ZtZ_copy); free(S_ztz); free(ZtU);
            }
        } else {
            // Pseudo-inverse
            solve_least_squares_double(Z, U_latent, B_latent, N, z_dim, target_dim);
        }
        printf("Regression completed.\n");

        if (!y_pca_mode && laplacian_lambda > 0) {
            printf("Applying laplacian smoothing...\n");
            double *b_cube = (double *)malloc(target_dim * sizeof(double)); // target_dim == P_Y == xa_y * ya_y
            double *b_cube_out = (double *)malloc(target_dim * sizeof(double));

            for (int i = 0; i < z_dim; i++) {
                // Extracts spatial frame from B_latent: B_latent is z_dim x target_dim. So i-th row is a frame.
                for (int j = 0; j < target_dim; j++) b_cube[j] = B_latent[i * target_dim + j];
                for (int iter = 0; iter < 10; iter++) {
                    apply_laplacian_double(b_cube, b_cube_out, xa_y, ya_y);
                    for (int j = 0; j < target_dim; j++) b_cube[j] -= laplacian_lambda * b_cube_out[j];
                }
                for (int j = 0; j < target_dim; j++) B_latent[i * target_dim + j] = b_cube[j];
            }
            free(b_cube); free(b_cube_out);
        }

        // Save everything
        char path[1024];
        snprintf(path, 1024, "%s_B_latent.fits", out_prefix);
        write_fits_2d(path, B_latent, target_dim, z_dim); 

        snprintf(path, 1024, "%s_PCx.fits", out_prefix);
        write_fits_2d(path, PCx, nx, P_X); 

        snprintf(path, 1024, "%s_PCy_local.fits", out_prefix);
        if (y_pca_mode) write_fits_2d(path, PCy_global, (long)patch_pixels * ny_per_patch, (long)n_patches);

        snprintf(path, 1024, "%s_Xmean.fits", out_prefix);
        write_fits_2d(path, X_mean, P_X, 1);
        snprintf(path, 1024, "%s_Xstd.fits", out_prefix);
        write_fits_2d(path, X_std, P_X, 1);

        snprintf(path, 1024, "%s_Ymean.fits", out_prefix);
        if (nax_y == 3) {
            write_fits_3d(path, Y_mean, xa_y, ya_y, 1);
            snprintf(path, 1024, "%s_Ystd.fits", out_prefix);
            write_fits_3d(path, Y_std, xa_y, ya_y, 1);
        } else {
            write_fits_2d(path, Y_mean, P_Y, 1);
            snprintf(path, 1024, "%s_Ystd.fits", out_prefix);
            write_fits_2d(path, Y_std, P_Y, 1);
        }

        printf("Output FITS files generated successfully with prefix: %s\n", out_prefix);

        snprintf(path, 1024, "%s_v2_info.txt", out_prefix);
        FILE *fp = fopen(path, "w");
        fprintf(fp, "patchsize %d\n", patchsize);
        fprintf(fp, "ny_per_patch %d\n", ny_per_patch);
        fprintf(fp, "nxp %d\n", nxp);
        fprintf(fp, "nyp %d\n", nyp);
        fprintf(fp, "y_pca_mode %d\n", y_pca_mode);
        fclose(fp);

        free(X); free(Y); free(X_mean); free(Y_mean); free(X_std); free(Y_std);
        free(Xc_copy); free(S_x); free(U_x); free(Vt_x); free(PCx); free(T); 
        free(Z); free(B_latent);
        if (y_pca_mode) free(PCy_global);
        free(U_latent);
    } else {
        // ===================================
        // SINGLE PRECISION PATH
        // ===================================
        float *X = NULL, *Y = NULL;
        long N_X, P_X, N_Y, P_Y;
        int xa_x, ya_x, xa_y, ya_y, nax_x, nax_y;

        read_fits_float(x_file, &X, &N_X, &P_X, &xa_x, &ya_x, &nax_x);
        read_fits_float(y_file, &Y, &N_Y, &P_Y, &xa_y, &ya_y, &nax_y);

        int N = (train_size > 0 && train_size < N_X) ? train_size : N_X;
        if (N > N_Y) N = N_Y;

        printf("Using N = %d samples. P_X = %ld, P_Y = %ld\n", N, P_X, P_Y);

        float *X_mean = (float*)calloc(P_X, sizeof(float));
        float *Y_mean = (float*)calloc(P_Y, sizeof(float));
        float *X_std = (float*)malloc(P_X * sizeof(float));
        float *Y_std = (float*)malloc(P_Y * sizeof(float));

        for(long j=0; j<P_X; j++) X_std[j] = 1.0f;
        for(long j=0; j<P_Y; j++) Y_std[j] = 1.0f;

        for (int i = 0; i < N; i++) {
            for (long j = 0; j < P_X; j++) X_mean[j] += X[i * P_X + j];
            for (long j = 0; j < P_Y; j++) Y_mean[j] += Y[i * P_Y + j];
        }
        for (long j = 0; j < P_X; j++) X_mean[j] /= N;
        for (long j = 0; j < P_Y; j++) Y_mean[j] /= N;

        for (int i = 0; i < N; i++) {
            for (long j = 0; j < P_X; j++) X[i * P_X + j] -= X_mean[j];
            for (long j = 0; j < P_Y; j++) Y[i * P_Y + j] -= Y_mean[j];
        }

        if (scale) {
            for (long i = 0; i < P_X; i++) X_std[i] = 0;
            for (long i = 0; i < P_Y; i++) Y_std[i] = 0;

            for (int i = 0; i < N; i++) {
                for (long j = 0; j < P_X; j++) {
                    float v = X[i * P_X + j];
                    X_std[j] += v * v;
                }
                for (long j = 0; j < P_Y; j++) {
                    float v = Y[i * P_Y + j];
                    Y_std[j] += v * v;
                }
            }
            for (long j = 0; j < P_X; j++) {
                X_std[j] = sqrtf(X_std[j] / (N - 1));
                if (X_std[j] < 1e-12f) X_std[j] = 1e-12f;
            }
            for (long j = 0; j < P_Y; j++) {
                Y_std[j] = sqrtf(Y_std[j] / (N - 1));
                if (Y_std[j] < 1e-12f) Y_std[j] = 1e-12f;
            }

            for (int i = 0; i < N; i++) {
                for (long j = 0; j < P_X; j++) X[i * P_X + j] /= X_std[j];
                for (long j = 0; j < P_Y; j++) Y[i * P_Y + j] /= Y_std[j];
            }
        }

        printf("PCA on X...\n");
        float *Xc_copy = (float *)malloc(N * P_X * sizeof(float));
        memcpy(Xc_copy, X, N * P_X * sizeof(float));

        int min_dim_X = N < P_X ? N : P_X;
        float *S_x = (float *)malloc(min_dim_X * sizeof(float));
        float *U_x = (float *)malloc(N * min_dim_X * sizeof(float));
        float *Vt_x = (float *)malloc(min_dim_X * P_X * sizeof(float));
        LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'S', N, P_X, Xc_copy, P_X, S_x, U_x, min_dim_X, Vt_x, P_X);

        if (nx == -1) {
            float sum_var = 0;
            for(int k=0; k<min_dim_X; k++) sum_var += (S_x[k]*S_x[k]) / (N-1);
            float cum_var = 0;
            nx = min_dim_X;
            for(int k=0; k<min_dim_X; k++) {
                cum_var += (S_x[k]*S_x[k]) / (N-1);
                if (cum_var / sum_var >= 0.95f) {
                    nx = k + 1;
                    break;
                }
            }
            printf("WFS PC number set to %d\n", nx);
        }
        if (nx > min_dim_X) nx = min_dim_X;
        printf("PCA on X completed.\n");

        float *PCx = (float *)malloc(P_X * nx * sizeof(float));
        for (long p = 0; p < P_X; p++) {
            for (int k = 0; k < nx; k++) {
                PCx[p * nx + k] = Vt_x[k * P_X + p];
            }
        }

        float *T = (float *)malloc(N * nx * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, nx, P_X, 1.0f, X, P_X, PCx, nx, 0.0f, T, nx);

        float *PCy_global = NULL; 
        float *U_latent = NULL;
        int target_dim_total = 0;
        int nxp = xa_y / patchsize;
        int nyp = ya_y / patchsize;
        int n_patches = nxp * nyp;
        int patch_pixels = patchsize * patchsize;
        int ny_per_patch = (ny <= 0) ? patch_pixels : ny;
        int y_pca_mode = (ny > 0);

        printf("Local processing on PSF (float): %d patches (%dx%d), ny=%d (%s)\n", 
               n_patches, patchsize, patchsize, ny_per_patch, y_pca_mode ? "PCA" : "Raw Pixels");
        
        target_dim_total = n_patches * ny_per_patch;
        U_latent = (float *)malloc(N * target_dim_total * sizeof(float));
        if (y_pca_mode) PCy_global = (float *)malloc(n_patches * patch_pixels * ny_per_patch * sizeof(float));

        for (int py = 0; py < nyp; py++) {
            for (int px = 0; px < nxp; px++) {
                int patch_idx = py * nxp + px;
                float *Y_patch = (float *)malloc(N * patch_pixels * sizeof(float));
                for (int i = 0; i < N; i++) {
                    for (int dy = 0; dy < patchsize; dy++) {
                        for (int dx = 0; dx < patchsize; dx++) {
                            Y_patch[i * patch_pixels + dy * patchsize + dx] = Y[i * P_Y + (py * patchsize + dy) * xa_y + (px * patchsize + dx)];
                        }
                    }
                }

                if (y_pca_mode) {
                    float *Yc_patch = (float *)malloc(N * patch_pixels * sizeof(float));
                    memcpy(Yc_patch, Y_patch, N * patch_pixels * sizeof(float));
                    int min_dim_patch = N < patch_pixels ? N : patch_pixels;
                    float *S_p = (float *)malloc(min_dim_patch * sizeof(float));
                    float *U_p = (float *)malloc(N * min_dim_patch * sizeof(float));
                    float *Vt_p = (float *)malloc(min_dim_patch * patch_pixels * sizeof(float));
                    LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'S', N, patch_pixels, Yc_patch, patch_pixels, S_p, U_p, min_dim_patch, Vt_p, patch_pixels);

                    int ny_this = ny_per_patch > min_dim_patch ? min_dim_patch : ny_per_patch;
                    for (int p = 0; p < patch_pixels; p++) {
                        for (int k = 0; k < ny_this; k++) {
                            PCy_global[patch_idx * (patch_pixels * ny_per_patch) + p * ny_per_patch + k] = Vt_p[k * patch_pixels + p];
                        }
                    }

                    float *PCy_this = &PCy_global[patch_idx * (patch_pixels * ny_per_patch)];
                    float *U_this = (float *)malloc(N * ny_per_patch * sizeof(float));
                    memset(U_this, 0, N * ny_per_patch * sizeof(float));
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, ny_this, patch_pixels, 1.0f, Y_patch, patch_pixels, PCy_this, ny_per_patch, 0.0f, U_this, ny_per_patch);

                    for (int i = 0; i < N; i++) {
                        for (int k = 0; k < ny_per_patch; k++) {
                            U_latent[i * target_dim_total + patch_idx * ny_per_patch + k] = U_this[i * ny_per_patch + k];
                        }
                    }
                    free(Yc_patch); free(S_p); free(U_p); free(Vt_p); free(U_this);
                } else {
                    for (int i = 0; i < N; i++) {
                        for (int k = 0; k < patch_pixels; k++) {
                            U_latent[i * target_dim_total + patch_idx * patch_pixels + k] = Y_patch[i * patch_pixels + k];
                        }
                    }
                }
                free(Y_patch);
            }
        }
        int target_dim = target_dim_total;

        int z_dim = use_quadratic ? (nx + nx*(nx+1)/2) : nx;
        float *Z = (float *)malloc(N * z_dim * sizeof(float));
        if (!Z) { fprintf(stderr, "Failed to allocate Z (N=%d, z_dim=%d)\n", (int)N, z_dim); exit(1); }
        if (use_quadratic) quadratic_expand_float(T, Z, N, nx);
        else memcpy(Z, T, N * nx * sizeof(float));

        if (noise_std > 0) {
            for (int i = 0; i < N * z_dim; i++) Z[i] += (float)(rand_normal() * noise_std);
        }

        float *B_latent = (float *)malloc(z_dim * target_dim * sizeof(float));
        if (reg_mode) {
            float eps = FLT_EPSILON;
            if (N < z_dim) {
                float *ZZt = (float *)malloc(N * N * sizeof(float));
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, z_dim, 1.0f, Z, z_dim, Z, z_dim, 0.0f, ZZt, N);
                float *ZZt_copy = (float *)malloc(N * N * sizeof(float));
                memcpy(ZZt_copy, ZZt, N * N * sizeof(float));
                float *S_zzt = (float *)malloc(N * sizeof(float));
                LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'N', N, N, ZZt_copy, N, S_zzt, NULL, 1, NULL, 1);
                float ridge_auto = eps * z_dim * S_zzt[0];
                for (int i = 0; i < N; i++) ZZt[i * N + i] += ridge_auto;
                float *M = (float *)malloc(N * target_dim * sizeof(float));
                memcpy(M, U_latent, N * target_dim * sizeof(float));
                LAPACKE_sposv(LAPACK_ROW_MAJOR, 'U', N, target_dim, ZZt, N, M, target_dim);
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, z_dim, target_dim, N, 1.0f, Z, z_dim, M, target_dim, 0.0f, B_latent, target_dim);
                free(ZZt); free(ZZt_copy); free(S_zzt); free(M);
            } else {
                float *ZtZ = (float *)malloc(z_dim * z_dim * sizeof(float));
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, z_dim, z_dim, N, 1.0f, Z, z_dim, Z, z_dim, 0.0f, ZtZ, z_dim);
                float *ZtZ_copy = (float *)malloc(z_dim * z_dim * sizeof(float));
                memcpy(ZtZ_copy, ZtZ, z_dim * z_dim * sizeof(float));
                float *S_ztz = (float *)malloc(z_dim * sizeof(float));
                LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'N', z_dim, z_dim, ZtZ_copy, z_dim, S_ztz, NULL, 1, NULL, 1);
                float ridge_auto = eps * z_dim * S_ztz[0];
                for (int i = 0; i < z_dim; i++) ZtZ[i * z_dim + i] += ridge_auto;
                float *ZtU = (float *)malloc(z_dim * target_dim * sizeof(float));
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, z_dim, target_dim, N, 1.0f, Z, z_dim, U_latent, target_dim, 0.0f, ZtU, target_dim);
                LAPACKE_sposv(LAPACK_ROW_MAJOR, 'U', z_dim, target_dim, ZtZ, z_dim, ZtU, target_dim);
                memcpy(B_latent, ZtU, z_dim * target_dim * sizeof(float));
                free(ZtZ); free(ZtZ_copy); free(S_ztz); free(ZtU);
            }
        } else {
            solve_least_squares_float(Z, U_latent, B_latent, N, z_dim, target_dim);
        }

        if (laplacian_lambda > 0) {
            float *b_cube = (float *)malloc(target_dim * sizeof(float));
            float *b_cube_out = (float *)malloc(target_dim * sizeof(float));
            for (int i = 0; i < z_dim; i++) {
                for (int j = 0; j < target_dim; j++) b_cube[j] = B_latent[i * target_dim + j];
                // Note: Laplacian on patch coefficients might be less meaningful geometrically,
                // but we can try applying it if target_dim was spatial.
                // However, here target_dim is concatenated coefficients. 
                // geometry-aware smoothing would be harder here. 
                // For now skip or apply dummy. 
            }
            free(b_cube); free(b_cube_out);
        }

        char path[1024];
        snprintf(path, 1024, "%s_B_latent.fits", out_prefix);
        write_fits_2d_float(path, B_latent, target_dim, z_dim); 

        snprintf(path, 1024, "%s_PCx.fits", out_prefix);
        write_fits_2d_float(path, PCx, nx, P_X); 

        snprintf(path, 1024, "%s_PCy_local.fits", out_prefix);
        write_fits_2d_float(path, PCy_global, (long)patch_pixels * ny_per_patch, (long)n_patches);

        snprintf(path, 1024, "%s_Xmean.fits", out_prefix);
        write_fits_2d_float(path, X_mean, P_X, 1);
        snprintf(path, 1024, "%s_Xstd.fits", out_prefix);
        write_fits_2d_float(path, X_std, P_X, 1);

        snprintf(path, 1024, "%s_Ymean.fits", out_prefix);
        if (nax_y == 3) write_fits_3d_float(path, Y_mean, xa_y, ya_y, 1);
        else write_fits_2d_float(path, Y_mean, P_Y, 1);
        snprintf(path, 1024, "%s_Ystd.fits", out_prefix);
        if (nax_y == 3) write_fits_3d_float(path, Y_std, xa_y, ya_y, 1);
        else write_fits_2d_float(path, Y_std, P_Y, 1);

        snprintf(path, 1024, "%s_v2_info.txt", out_prefix);
        FILE *fp = fopen(path, "w");
        fprintf(fp, "patchsize %d\n", patchsize);
        fprintf(fp, "ny_per_patch %d\n", ny_per_patch);
        fprintf(fp, "nxp %d\n", nxp);
        fprintf(fp, "nyp %d\n", nyp);
        fclose(fp);

        free(X); free(Y); free(X_mean); free(Y_mean); free(X_std); free(Y_std);
        free(Xc_copy); free(S_x); free(U_x); free(Vt_x); free(PCx); free(T); 
        free(Z); free(B_latent); free(PCy_global); free(U_latent);
    }

    return 0;
}
