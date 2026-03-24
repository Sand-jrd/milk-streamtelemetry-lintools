#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <float.h>
#include "common.h"

// Helper to expand T to Z (quadratic)
void quadratic_expand_double(const double *T, double *Z, long n_samples, int nx) {
    int z_dim = 1 + nx + nx * (nx + 1) / 2;
    for (long i = 0; i < n_samples; i++) {
        const double *t = &T[i * nx];
        double *z = &Z[i * z_dim];
        int idx = 0;
        z[idx++] = 1.0; // Bias term
        for (int j = 0; j < nx; j++)
            z[idx++] = t[j];
        for (int j = 0; j < nx; j++)
            for (int k = j; k < nx; k++)
                z[idx++] = t[j] * t[k];
    }
}

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
    printf("  -noscale              Disable X scaling (default: enabled)\n");
    printf("  -noquad               Disable quadratic expansion (default: enabled)\n");
    printf("  -noise <float>        Noise std (default: 1e-9)\n");
    printf("  -trainsize <int>      Training set size (default: all)\n");
    printf("  -ridge <float>        Ridge lambda (default: auto from spectral norm)\n");
    printf("  -autosmooth <float>   Semi-auto smooth: multiplier on training roughness (default: 0)\n");
    printf("  -epsilon <float>      Positivity tolerance: ZB >= -epsilon (default: -1 = disabled)\n");
    printf("  -rho <float>          ADMM penalty rho (default: 1.0)\n");
    printf("  -admm_iters <int>     ADMM iterations (default: 50)\n");
}

/*
 * ADMM QP Solver:
 *   min  ||ZB - Y||_F^2 + lambda * ||B||_F^2
 *   s.t. ZB >= -epsilon  (element-wise on training set)
 *
 * Split: S = ZB
 *   min  ||S - Y||_F^2 + lambda * ||B||_F^2
 *   s.t. ZB = S, S >= -epsilon
 *
 * ADMM updates (scaled dual U):
 *   B <- solve (lambda*I + rho*Z^TZ) B = rho * Z^T(S - U)
 *   S <- max( (2Y + rho*(ZB + U)) / (2+rho), -epsilon )
 *   U <- U + ZB - S
 *
 * B update uses precomputed Cholesky factor of (lambda*I + rho*Z^TZ).
 */
// 2D Spatial Smoothing proximal (Laplacian-like)
static void spatial_smooth_patch(double *b_map, int patchsize, double strength) {
    if (strength <= 0) return;
    double *tmp = (double *)malloc(patchsize * patchsize * sizeof(double));
    memcpy(tmp, b_map, patchsize * patchsize * sizeof(double));

    // Simple 3x3 Laplacian-style smoothing: b = (1-w)*b + w*blur(b)
    // w is related to 'strength'
    double w = strength / (1.0 + strength);
    if (w > 0.8) w = 0.8; // cap to prevent excessive blurring

    for (int y = 0; y < patchsize; y++) {
        for (int x = 0; x < patchsize; x++) {
            double sum = 0;
            int count = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;
                    if (ny >= 0 && ny < patchsize && nx >= 0 && nx < patchsize) {
                        sum += tmp[ny * patchsize + nx];
                        count++;
                    }
                }
            }
            b_map[y * patchsize + x] = (1.0 - w) * tmp[y * patchsize + x] + w * (sum / count);
        }
    }
    free(tmp);
}

// Estimate the intrinsic roughness of the training data
static double estimate_roughness_double(const double *Y, int N, long P_Y, int xa, int ya) {
    int N_check = N > 50 ? 50 : N; // sample up to 50 frames
    double total_r = 0;
    for (int i = 0; i < N_check; i++) {
        const double *img = &Y[i * P_Y];
        double frame_r = 0;
        double frame_energy = 0;
        for (int y = 1; y < ya - 1; y++) {
            for (int x = 1; x < xa - 1; x++) {
                double val = img[y * xa + x];
                double lap = 4.0 * val - (img[(y-1)*xa + x] + img[(y+1)*xa + x] + img[y*xa + (x-1)] + img[y*xa + (x+1)]);
                frame_r += lap * lap;
                frame_energy += val * val;
            }
        }
        if (frame_energy > 1e-12) total_r += sqrt(frame_r / frame_energy);
    }
    return total_r / N_check;
}

static void admm_qp_double(
    const double *Z,      // N x z_dim
    const double *Y_lat,  // N x target_dim  (residual U_latent)
    double *B,            // z_dim x target_dim  (output)
    int N, int z_dim, long target_dim,
    double lambda, double epsilon, double rho, int admm_iters,
    const double *Y_mean_patch, // Mean of Y in this patch/basis
    int y_pca_mode,
    double smooth_strength, int patchsize)
{
    printf("ADMM QP: N=%d z_dim=%d target_dim=%ld lambda=%g eps=%g rho=%g iters=%d (PCA=%d)\n",
           N, z_dim, target_dim, lambda, epsilon, rho, admm_iters, y_pca_mode);

    // --- Precompute Z^T Z ---
    double *ZtZ = (double *)malloc((long)z_dim * z_dim * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                z_dim, z_dim, N, 1.0, Z, z_dim, Z, z_dim, 0.0, ZtZ, z_dim);

    // --- Cholesky factor of A_B = lambda*I + rho*Z^TZ ---
    double *A_B = (double *)malloc((long)z_dim * z_dim * sizeof(double));
    for (long i = 0; i < (long)z_dim * z_dim; i++) A_B[i] = rho * ZtZ[i];
    for (int i = 0; i < z_dim; i++) A_B[i * z_dim + i] += lambda;
    if (LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', z_dim, A_B, z_dim) != 0) {
        fprintf(stderr, "ADMM: Cholesky of A_B failed, falling back to ridge.\n");
        // Fallback: standard ridge
        double *ZtU = (double *)malloc((long)z_dim * target_dim * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    z_dim, (int)target_dim, N, 1.0, Z, z_dim, Y_lat, (int)target_dim, 0.0, ZtU, (int)target_dim);
        double *Areg = (double *)malloc((long)z_dim * z_dim * sizeof(double));
        memcpy(Areg, ZtZ, (long)z_dim * z_dim * sizeof(double));
        for (int i = 0; i < z_dim; i++) Areg[i * z_dim + i] += lambda;
        LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', z_dim, Areg, z_dim);
        LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'U', z_dim, (int)target_dim, Areg, z_dim, ZtU, (int)target_dim);
        memcpy(B, ZtU, (long)z_dim * target_dim * sizeof(double));
        free(ZtU); free(Areg); free(ZtZ); free(A_B);
        return;
    }

    // --- Initialize B from unconstrained ridge solution ---
    {
        double *ZtU_init = (double *)malloc((long)z_dim * target_dim * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    z_dim, (int)target_dim, N, 1.0, Z, z_dim, Y_lat, (int)target_dim, 0.0, ZtU_init, (int)target_dim);
        double *A_init = (double *)malloc((long)z_dim * z_dim * sizeof(double));
        memcpy(A_init, ZtZ, (long)z_dim * z_dim * sizeof(double));
        for (int i = 0; i < z_dim; i++) A_init[i * z_dim + i] += lambda;
        LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', z_dim, A_init, z_dim);
        LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'U', z_dim, (int)target_dim, A_init, z_dim, ZtU_init, (int)target_dim);
        memcpy(B, ZtU_init, (long)z_dim * target_dim * sizeof(double));
        free(ZtU_init); free(A_init);
    }

    // --- Allocate ADMM workspace ---
    long NxT = (long)N * target_dim;
    double *ZB      = (double *)malloc(NxT * sizeof(double));
    double *S       = (double *)malloc(NxT * sizeof(double));
    double *U_admm  = (double *)calloc(NxT, sizeof(double));
    double *RHS     = (double *)malloc((long)z_dim * target_dim * sizeof(double));

    // Initialize S = clip(ZB_init, -epsilon, inf)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, (int)target_dim, z_dim, 1.0, Z, z_dim, B, (int)target_dim, 0.0, ZB, (int)target_dim);
    for (long k = 0; k < NxT; k++) S[k] = ZB[k] < -epsilon ? -epsilon : ZB[k];

    double denom = 2.0 + rho;

    for (int iter = 0; iter < admm_iters; iter++) {
        // --- B update: solve (lambda*I + rho*Z^TZ) B = rho * Z^T(S - U) ---
        // Compute Z^T(S - U) into RHS
        // Temp buffer: subtract U from S in-place temporarily, then restore
        for (long k = 0; k < NxT; k++) ZB[k] = S[k] - U_admm[k]; // reuse ZB as temp
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    z_dim, (int)target_dim, N, rho, Z, z_dim, ZB, (int)target_dim, 0.0, RHS, (int)target_dim);
        // Solve using precomputed Cholesky
        memcpy(B, RHS, (long)z_dim * target_dim * sizeof(double));
        LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'U', z_dim, (int)target_dim, A_B, z_dim, B, (int)target_dim);

        // --- New: Spatial Smoothing Proxy ---
        if (smooth_strength > 0 && !y_pca_mode) {
            for (int k = 0; k < z_dim; k++) {
                spatial_smooth_patch(&B[k * target_dim], patchsize, smooth_strength);
            }
        }

        // --- Compute ZB ---
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, (int)target_dim, z_dim, 1.0, Z, z_dim, B, (int)target_dim, 0.0, ZB, (int)target_dim);

        // --- S update ---
        for (long k = 0; k < NxT; k++) {
            double s_unc = (2.0 * Y_lat[k] + rho * (ZB[k] + U_admm[k])) / denom;
            double bound = -epsilon;
            if (!y_pca_mode) {
                bound = -Y_mean_patch[k % target_dim] - epsilon;
            }
            S[k] = s_unc < bound ? bound : s_unc;
        }

        // --- U update: U = U + ZB - S ---
        for (long k = 0; k < NxT; k++) U_admm[k] += ZB[k] - S[k];

        if ((iter + 1) % 10 == 0) {
            // Compute primal residual norm for monitoring
            double res = 0;
            for (long k = 0; k < NxT; k++) {
                double r = ZB[k] - S[k];
                res += r * r;
            }
            printf("  ADMM iter %d/%d  primal_res=%.4e\n", iter+1, admm_iters, sqrt(res / NxT));
        }
    }

    free(ZtZ); free(A_B); free(ZB); free(S); free(U_admm); free(RHS);
}

int main(int argc, char **argv) {
    if (argc < 4) { print_help(argv[0]); return 1; }

    const char *x_file    = argv[1];
    const char *y_file    = argv[2];
    const char *out_prefix = argv[3];

    int    nx          = 10;
    int    ny          = 5;
    int    patchsize   = 16;
    int    scale       = 1;
    int    use_quadratic = 1;
    double noise_std   = 1e-9;
    int    train_size  = -1;
    double ridge_lambda = 0.0;   // 0 = auto
    double auto_smooth_mult = 0.0; // 0 = disabled
    double epsilon     = -1.0;   // <0 = disabled (standard ridge)
    double admm_rho    = 1.0;
    int    admm_iters  = 50;

    for (int i = 4; i < argc; i++) {
        if      (strcmp(argv[i], "-nx")         == 0) nx          = atoi(argv[++i]);
        else if (strcmp(argv[i], "-ny")         == 0) ny          = atoi(argv[++i]);
        else if (strcmp(argv[i], "-patchsize")  == 0) patchsize   = atoi(argv[++i]);
        else if (strcmp(argv[i], "-noscale")    == 0) scale       = 0;
        else if (strcmp(argv[i], "-noquad")     == 0) use_quadratic = 0;
        else if (strcmp(argv[i], "-noise")      == 0) noise_std   = atof(argv[++i]);
        else if (strcmp(argv[i], "-trainsize")  == 0) train_size  = atoi(argv[++i]);
        else if (strcmp(argv[i], "-ridge")      == 0) ridge_lambda = atof(argv[++i]);
        else if (strcmp(argv[i], "-autosmooth") == 0) auto_smooth_mult = atof(argv[++i]);
        else if (strcmp(argv[i], "-epsilon")    == 0) epsilon     = atof(argv[++i]);
        else if (strcmp(argv[i], "-rho")        == 0) admm_rho    = atof(argv[++i]);
        else if (strcmp(argv[i], "-admm_iters") == 0) admm_iters  = atoi(argv[++i]);
    }

    printf("v3 Settings: nx=%d ny=%d scale=%d quad=%d ridge=%g autosmooth=%g epsilon=%g rho=%g admm_iters=%d\n",
           nx, ny, scale, use_quadratic, ridge_lambda, auto_smooth_mult, epsilon, admm_rho, admm_iters);

    // ===== Double precision path =====
    double *X = NULL, *Y = NULL;
    long N_X, P_X, N_Y, P_Y;
    int xa_x, ya_x, xa_y, ya_y, nax_x, nax_y;

    read_fits(x_file, &X, &N_X, &P_X, &xa_x, &ya_x, &nax_x);
    read_fits(y_file, &Y, &N_Y, &P_Y, &xa_y, &ya_y, &nax_y);

    int N;
    if (train_size > 0) {
        if (train_size > (int)N_X || train_size > (int)N_Y) {
            fprintf(stderr, "Error: -trainsize %d is larger than available frames (N_X=%ld, N_Y=%ld)\n", train_size, N_X, N_Y);
            return 1;
        }
        N = train_size;
    } else {
        if (N_X != N_Y) {
            fprintf(stderr, "Error: Frame count mismatch! X has %ld frames but Y has %ld frames.\n", N_X, N_Y);
            fprintf(stderr, "Telemetery and Images must be synchronized and have the same length.\n");
            return 1;
        }
        N = (int)N_X;
    }
    printf("Using N=%d samples for training. P_X=%ld P_Y=%ld\n", N, P_X, P_Y);
    // --- Mean/Std for X only (Y is NOT normalized in QP mode) ---
    double *X_mean = (double *)calloc(P_X, sizeof(double));
    double *Y_mean = (double *)calloc(P_Y, sizeof(double));
    double *X_std  = (double *)malloc(P_X * sizeof(double));
    double *Y_std  = (double *)malloc(P_Y * sizeof(double));
    for (long j = 0; j < P_X; j++) X_std[j] = 1.0;
    for (long j = 0; j < P_Y; j++) Y_std[j] = 1.0;

    // Compute X mean
    for (int i = 0; i < N; i++)
        for (long j = 0; j < P_X; j++) X_mean[j] += X[i * P_X + j];
    for (long j = 0; j < P_X; j++) X_mean[j] /= N;

    // Compute Y mean (saved for prediction, but NOT subtracted from Y in QP mode)
    for (int i = 0; i < N; i++)
        for (long j = 0; j < P_Y; j++) Y_mean[j] += Y[i * P_Y + j];
    for (long j = 0; j < P_Y; j++) Y_mean[j] /= N;

    // Subtract X mean
    for (int i = 0; i < N; i++)
        for (long j = 0; j < P_X; j++) X[i * P_X + j] -= X_mean[j];

    // Subtract Y mean (Always do this, even in QP mode, so model learns residuals)
    for (int i = 0; i < N; i++)
        for (long j = 0; j < P_Y; j++)
            Y[i * P_Y + j] -= Y_mean[j];
    // In QP/positivity mode: Y is NOT mean-subtracted (not divided by std either)

    // Scale X
    if (scale) {
        for (long i = 0; i < P_X; i++) X_std[i] = 0;
        for (int i = 0; i < N; i++)
            for (long j = 0; j < P_X; j++) { double v = X[i*P_X+j]; X_std[j] += v*v; }
        for (long j = 0; j < P_X; j++) {
            X_std[j] = sqrt(X_std[j] / (N - 1));
            if (X_std[j] < 1e-12) X_std[j] = 1e-12;
        }
        for (int i = 0; i < N; i++)
            for (long j = 0; j < P_X; j++) X[i*P_X+j] /= X_std[j];
    }

    // --- PCA on X ---
    printf("PCA on X...\n");
    double *Xc_copy = (double *)malloc((long)N * P_X * sizeof(double));
    memcpy(Xc_copy, X, (long)N * P_X * sizeof(double));
    int min_dim_X = N < (int)P_X ? N : (int)P_X;
    double *S_x  = (double *)malloc(min_dim_X * sizeof(double));
    double *U_x  = (double *)malloc((long)N * min_dim_X * sizeof(double));
    double *Vt_x = (double *)malloc((long)min_dim_X * P_X * sizeof(double));
    LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', N, (int)P_X, Xc_copy, (int)P_X,
                   S_x, U_x, min_dim_X, Vt_x, (int)P_X);

    if (nx == -1) {
        double sum_var = 0, cum_var = 0;
        for (int k = 0; k < min_dim_X; k++) sum_var += S_x[k] * S_x[k];
        nx = min_dim_X;
        for (int k = 0; k < min_dim_X; k++) {
            cum_var += S_x[k] * S_x[k];
            if (cum_var / sum_var >= 0.95) { nx = k + 1; break; }
        }
        printf("Auto nx=%d\n", nx);
    }
    if (nx > min_dim_X) nx = min_dim_X;
    printf("PCA on X done (nx=%d).\n", nx);

    double *PCx = (double *)malloc((long)P_X * nx * sizeof(double));
    for (long p = 0; p < P_X; p++)
        for (int k = 0; k < nx; k++)
            PCx[p * nx + k] = Vt_x[k * P_X + p];

    double *T = (double *)malloc((long)N * nx * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, nx, (int)P_X, 1.0, X, (int)P_X, PCx, nx, 0.0, T, nx);

    // --- Local PCA on Y patches ---
    int nxp = xa_y / patchsize;
    int nyp = ya_y / patchsize;
    int n_patches   = nxp * nyp;
    int patch_pixels = patchsize * patchsize;
    int ny_per_patch = (ny <= 0) ? patch_pixels : ny;
    int y_pca_mode  = (ny > 0);
    long target_dim = (long)n_patches * ny_per_patch;

    printf("PSF patches: %d patches (%dx%d), ny=%d (%s)\n",
           n_patches, patchsize, patchsize, ny_per_patch, y_pca_mode ? "PCA" : "Raw");

    double *U_latent   = (double *)malloc((long)N * target_dim * sizeof(double));
    double *PCy_global = y_pca_mode ? (double *)malloc((long)n_patches * patch_pixels * ny_per_patch * sizeof(double)) : NULL;

    for (int py = 0; py < nyp; py++) {
        for (int px = 0; px < nxp; px++) {
            int patch_idx = py * nxp + px;
            double *Y_patch = (double *)malloc((long)N * patch_pixels * sizeof(double));
            for (int i = 0; i < N; i++)
                for (int dy = 0; dy < patchsize; dy++)
                    for (int dx = 0; dx < patchsize; dx++)
                        Y_patch[i*patch_pixels + dy*patchsize + dx] =
                            Y[i*P_Y + (py*patchsize+dy)*xa_y + (px*patchsize+dx)];

            if (y_pca_mode) {
                double *Yc = (double *)malloc((long)N * patch_pixels * sizeof(double));
                memcpy(Yc, Y_patch, (long)N * patch_pixels * sizeof(double));
                int min_dim_p = N < patch_pixels ? N : patch_pixels;
                double *S_p  = (double *)malloc(min_dim_p * sizeof(double));
                double *U_p  = (double *)malloc((long)N * min_dim_p * sizeof(double));
                double *Vt_p = (double *)malloc((long)min_dim_p * patch_pixels * sizeof(double));
                LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', N, patch_pixels, Yc, patch_pixels,
                               S_p, U_p, min_dim_p, Vt_p, patch_pixels);
                int ny_this = ny_per_patch > min_dim_p ? min_dim_p : ny_per_patch;
                for (int p = 0; p < patch_pixels; p++)
                    for (int k = 0; k < ny_this; k++)
                        PCy_global[patch_idx*(patch_pixels*ny_per_patch) + p*ny_per_patch + k] =
                            Vt_p[k*patch_pixels + p];

                double *PCy_this = &PCy_global[patch_idx*(patch_pixels*ny_per_patch)];
                double *U_this = (double *)calloc((long)N * ny_per_patch, sizeof(double));
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            N, ny_this, patch_pixels, 1.0,
                            Y_patch, patch_pixels, PCy_this, ny_per_patch,
                            0.0, U_this, ny_per_patch);
                for (int i = 0; i < N; i++)
                    for (int k = 0; k < ny_per_patch; k++)
                        U_latent[i*target_dim + patch_idx*ny_per_patch + k] = U_this[i*ny_per_patch + k];
                free(Yc); free(S_p); free(U_p); free(Vt_p); free(U_this);
            } else {
                for (int i = 0; i < N; i++)
                    for (int k = 0; k < patch_pixels; k++)
                        U_latent[i*target_dim + patch_idx*patch_pixels + k] =
                            Y_patch[i*patch_pixels + k];
            }
            free(Y_patch);
        }
    }
    printf("Local PCA on Y done. target_dim=%ld\n", target_dim);

    // --- Quadratic expansion ---
    int z_dim = use_quadratic ? (1 + nx + nx * (nx + 1) / 2) : nx;
    double *Z = (double *)malloc((long)N * z_dim * sizeof(double));
    if (!Z) { fprintf(stderr, "Failed to alloc Z\n"); exit(1); }
    printf("Quadratic expansion (z_dim=%d)...\n", z_dim);
    if (use_quadratic) quadratic_expand_double(T, Z, N, nx);
    else memcpy(Z, T, (long)N * nx * sizeof(double));

    if (noise_std > 0 && !y_pca_mode)
        for (long i = 0; i < (long)N * z_dim; i++) Z[i] += rand_normal() * noise_std;

    // --- Auto smooth ---
    double spatial_smooth = 0;
    if (auto_smooth_mult > 0) {
        double baseline_r = estimate_roughness_double(Y, N, P_Y, xa_y, ya_y);
        printf("Measured training roughness: %g\n", baseline_r);
        spatial_smooth = baseline_r * auto_smooth_mult;
        printf("Applying spatial_smooth = %g (multiplier %g)\n", spatial_smooth, auto_smooth_mult);
    }

    // --- Auto ridge lambda ---
    double lambda;
    if (ridge_lambda > 0) {
        lambda = ridge_lambda;
    } else {
        double *ZtZ_tmp = (double *)malloc((long)z_dim * z_dim * sizeof(double));
        double *ZtZ_sv  = (double *)malloc((long)z_dim * z_dim * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    z_dim, z_dim, N, 1.0, Z, z_dim, Z, z_dim, 0.0, ZtZ_tmp, z_dim);
        memcpy(ZtZ_sv, ZtZ_tmp, (long)z_dim * z_dim * sizeof(double));
        double *svals = (double *)malloc(z_dim * sizeof(double));
        LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'N', z_dim, z_dim, ZtZ_sv, z_dim, svals, NULL, 1, NULL, 1);
        lambda = DBL_EPSILON * z_dim * svals[0];
        printf("Auto ridge lambda = %g (spectral norm ZtZ = %g)\n", lambda, svals[0]);
        free(ZtZ_tmp); free(ZtZ_sv); free(svals);
    }

    // --- Regression ---
    double *B_latent = (double *)malloc((long)z_dim * target_dim * sizeof(double));
    if (!B_latent) { fprintf(stderr, "Failed to alloc B_latent\n"); exit(1); }

    printf("Starting regression (mode: %s)...\n", epsilon >= 0 ? "ADMM QP" : "Ridge");

    if (epsilon >= 0) {
        // ADMM QP with positivity constraint
        // Extract the part of Y_mean corresponding to the target_dim (patches)
        double *Y_mean_latent = (double *)calloc(target_dim, sizeof(double));
        if (y_pca_mode) {
            // In PCA mode, Y_mean is already accounted for globally. 
            // The coefficients U should be centered at 0.
            // Positivity constraint on coefficients alone is just U >= -epsilon.
        } else {
            // Raw pixels: map global Y_mean to the concatenated target_dim vector
            for (int py = 0; py < nyp; py++) {
                for (int px = 0; px < nxp; px++) {
                    int patch_idx = py * nxp + px;
                    for (int dy = 0; dy < patchsize; dy++) {
                        for (int dx = 0; dx < patchsize; dx++) {
                            Y_mean_latent[patch_idx * patch_pixels + dy * patchsize + dx] =
                                Y_mean[(py * patchsize + dy) * xa_y + (px * patchsize + dx)];
                        }
                    }
                }
            }
        }

        admm_qp_double(Z, U_latent, B_latent, N, z_dim, target_dim,
                       lambda, epsilon, admm_rho, admm_iters, Y_mean_latent, y_pca_mode,
                       spatial_smooth, patchsize);
        free(Y_mean_latent);
    } else {
        // Standard ridge (same as v2)
        double *ZtZ = (double *)malloc((long)z_dim * z_dim * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    z_dim, z_dim, N, 1.0, Z, z_dim, Z, z_dim, 0.0, ZtZ, z_dim);
        for (int i = 0; i < z_dim; i++) ZtZ[i*z_dim+i] += lambda;
        double *ZtU = (double *)malloc((long)z_dim * target_dim * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    z_dim, (int)target_dim, N, 1.0, Z, z_dim, U_latent, (int)target_dim, 0.0, ZtU, (int)target_dim);
        LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', z_dim, ZtZ, z_dim);
        LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'U', z_dim, (int)target_dim, ZtZ, z_dim, ZtU, (int)target_dim);
        memcpy(B_latent, ZtU, (long)z_dim * target_dim * sizeof(double));
        free(ZtZ); free(ZtU);
    }
    printf("Regression done.\n");

    // --- Save outputs ---
    char path[1024];
    snprintf(path, 1024, "%s_B_latent.fits", out_prefix);
    write_fits_2d(path, B_latent, target_dim, z_dim);
    snprintf(path, 1024, "%s_PCx.fits", out_prefix);
    write_fits_2d(path, PCx, nx, P_X);
    if (y_pca_mode) {
        snprintf(path, 1024, "%s_PCy_local.fits", out_prefix);
        write_fits_2d(path, PCy_global, (long)patch_pixels * ny_per_patch, (long)n_patches);
    }
    snprintf(path, 1024, "%s_Xmean.fits", out_prefix);
    write_fits_2d(path, X_mean, P_X, 1);
    snprintf(path, 1024, "%s_Xstd.fits", out_prefix);
    write_fits_2d(path, X_std, P_X, 1);
    snprintf(path, 1024, "%s_Ymean.fits", out_prefix);
    if (nax_y == 3) write_fits_3d(path, Y_mean, xa_y, ya_y, 1);
    else            write_fits_2d(path, Y_mean, P_Y, 1);
    snprintf(path, 1024, "%s_Ystd.fits", out_prefix);
    if (nax_y == 3) write_fits_3d(path, Y_std, xa_y, ya_y, 1);
    else            write_fits_2d(path, Y_std, P_Y, 1);

    // Write info file (v3 compatible, read by v3_predict)
    snprintf(path, 1024, "%s_v3_info.txt", out_prefix);
    FILE *fp = fopen(path, "w");
    fprintf(fp, "patchsize %d\n", patchsize);
    fprintf(fp, "ny_per_patch %d\n", ny_per_patch);
    fprintf(fp, "nxp %d\n", nxp);
    fprintf(fp, "nyp %d\n", nyp);
    fprintf(fp, "y_pca_mode %d\n", y_pca_mode);
    fprintf(fp, "epsilon %g\n", epsilon);
    fclose(fp);

    printf("v3 fit complete. Prefix: %s\n", out_prefix);

    free(X); free(Y); free(X_mean); free(Y_mean); free(X_std); free(Y_std);
    free(Xc_copy); free(S_x); free(U_x); free(Vt_x); free(PCx); free(T);
    free(Z); free(B_latent); free(U_latent);
    if (y_pca_mode) free(PCy_global);

    return 0;
}
