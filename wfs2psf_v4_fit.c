#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <float.h>
#include "common.h"

// Helper to expand T (just adds bias for v4 as the model is internally quadratic)
void expand_v4_double(const double *T, double *Z, long n_samples, int nx) {
    for (long i = 0; i < n_samples; i++) {
        Z[i * (nx + 1)] = 1.0; // Bias
        memcpy(&Z[i * (nx + 1) + 1], &T[i * nx], nx * sizeof(double));
    }
}

void print_help(const char *prog) {
    printf("Usage: %s <X_file.fits> <Y_file.fits> <out_prefix> [options]\n", prog);
    printf("Options:\n");
    printf("  -nx <int>             n_components_x (default: 10)\n");
    printf("  -ny <int>             n_components_y per patch (0 for raw pixels, recommended for v4)\n");
    printf("  -patchsize <int>      PSF patch size (default: 16)\n");
    printf("  -trainsize <int>      Training set size (default: all)\n");
    printf("  -iters <int>          GD iterations (default: 100)\n");
    printf("  -lr <float>           Learning rate (default: 0.1)\n");
    printf("  -lambda <float>       Regularization on B (default: 1e-6)\n");
}

/*
 * v4 Solver: Magnitude-only regression (Phase Retrieval style)
 *   min  sum_i || Y_i - |b^H X_i|^2 ||^2 + lambda * ||b||^2
 *
 * Each pixel k has its own complex vector b_k = b_re + i*b_im
 * Predicted intensity: y_pred = (b_re^T X)^2 + (b_im^T X)^2
 */
static void solve_v4_pixel(
    const double *X,      // N x nx_eff
    const double *Y_pix,  // N (intensities for ONE pixel)
    double *b_re,         // nx_eff (output)
    double *b_im,         // nx_eff (output)
    int N, int nx_eff,
    int iters, double lr, double lambda)
{
    // 1. Spectral Initialization: top eigenvector of sum_i Y_i * X_i X_i^T
    // For simplicity, initialize b_re with a scaled linear estimate and b_im small.
    double *Xty = (double *)calloc(nx_eff, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < nx_eff; j++) Xty[j] += Y_pix[i] * X[i * nx_eff + j];
    }
    double norm = 0;
    for (int j = 0; j < nx_eff; j++) norm += Xty[j] * Xty[j];
    norm = sqrt(norm + 1e-12);
    for (int j = 0; j < nx_eff; j++) {
        b_re[j] = 0.5 * Xty[j] / norm; 
        b_im[j] = 0.001 * ((double)rand()/RAND_MAX - 0.5);
    }
    free(Xty);

    // 2. Gradient Descent
    double *pred = (double *)malloc(N * sizeof(double));
    double *score_re = (double *)malloc(N * sizeof(double));
    double *score_im = (double *)malloc(N * sizeof(double));

    for (int iter = 0; iter < iters; iter++) {
        // Compute current predictions and gradients
        cblas_dgemv(CblasRowMajor, CblasNoTrans, N, nx_eff, 1.0, X, nx_eff, b_re, 1, 0.0, score_re, 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, N, nx_eff, 1.0, X, nx_eff, b_im, 1, 0.0, score_im, 1);

        for (int i = 0; i < N; i++) {
            pred[i] = score_re[i] * score_re[i] + score_im[i] * score_im[i];
        }

        // Gradients: grad_re = sum_i 4 * (pred_i - Y_i) * score_re_i * X_i + 2 * lambda * b_re
        double *grad_re = (double *)calloc(nx_eff, sizeof(double));
        double *grad_im = (double *)calloc(nx_eff, sizeof(double));

        for (int i = 0; i < N; i++) {
            double diff = 4.0 * (pred[i] - Y_pix[i]);
            cblas_daxpy(nx_eff, diff * score_re[i], &X[i * nx_eff], 1, grad_re, 1);
            cblas_daxpy(nx_eff, diff * score_im[i], &X[i * nx_eff], 1, grad_im, 1);
        }

        // Apply steps
        for (int j = 0; j < nx_eff; j++) {
            b_re[j] -= lr * (grad_re[j] / N + 2.0 * lambda * b_re[j]);
            b_im[j] -= lr * (grad_im[j] / N + 2.0 * lambda * b_im[j]);
        }

        free(grad_re); free(grad_im);
    }

    free(pred); free(score_re); free(score_im);
}

int main(int argc, char **argv) {
    if (argc < 4) { print_help(argv[0]); return 1; }

    const char *x_file     = argv[1];
    const char *y_file     = argv[2];
    const char *out_prefix = argv[3];

    int nx           = 10;
    int ny           = 0; // v4 default is raw pixels
    int patchsize    = 16;
    int train_size   = -1;
    int max_iters    = 100;
    double lr        = 0.1;
    double reg_lambda = 1e-6;

    for (int i = 4; i < argc; i++) {
        if      (strcmp(argv[i], "-nx")        == 0) nx         = atoi(argv[++i]);
        else if (strcmp(argv[i], "-ny")        == 0) ny         = atoi(argv[++i]);
        else if (strcmp(argv[i], "-patchsize") == 0) patchsize  = atoi(argv[++i]);
        else if (strcmp(argv[i], "-trainsize") == 0) train_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "-iters")     == 0) max_iters  = atoi(argv[++i]);
        else if (strcmp(argv[i], "-lr")        == 0) lr         = atof(argv[++i]);
        else if (strcmp(argv[i], "-lambda")    == 0) reg_lambda = atof(argv[++i]);
    }

    double *X_raw = NULL, *Y_raw = NULL;
    long N_X, P_X, N_Y, P_Y;
    int xa_x, ya_x, xa_y, ya_y, nax_x, nax_y;
    read_fits(x_file, &X_raw, &N_X, &P_X, &xa_x, &ya_x, &nax_x);
    read_fits(y_file, &Y_raw, &N_Y, &P_Y, &xa_y, &ya_y, &nax_y);

    int N = (train_size > 0 && train_size < N_X) ? train_size : (int)N_X;
    if (N > (int)N_Y) N = (int)N_Y;

    // 1. Center X
    double *X_mean = (double *)calloc(P_X, sizeof(double));
    double *X_std  = (double *)malloc(P_X * sizeof(double));
    for (int i = 0; i < N; i++)
        for (long j = 0; j < P_X; j++) X_mean[j] += X_raw[i * P_X + j];
    for (long j = 0; j < P_X; j++) X_mean[j] /= N;

    for (long j = 0; j < P_X; j++) X_std[j] = 0;
    for (int i = 0; i < N; i++)
        for (long j = 0; j < P_X; j++) {
            double v = X_raw[i*P_X+j] - X_mean[j];
            X_std[j] += v*v;
        }
    for (long j = 0; j < P_X; j++) {
        X_std[j] = sqrt(X_std[j] / (N - 1));
        if (X_std[j] < 1e-12) X_std[j] = 1e-12;
    }
    
    for (int i = 0; i < N; i++)
        for (long j = 0; j < P_X; j++) X_raw[i * P_X + j] = (X_raw[i * P_X + j] - X_mean[j]) / X_std[j];

    // 2. PCA on X
    printf("PCA on X (nx=%d)...\n", nx);
    double *Xc = (double *)malloc((long)N * P_X * sizeof(double));
    memcpy(Xc, X_raw, (long)N * P_X * sizeof(double));
    double *Sx = (double *)malloc(nx * sizeof(double));
    double *Ux = (double *)malloc((long)N * nx * sizeof(double));
    double *Vtx = (double *)malloc((long)nx * P_X * sizeof(double));
    LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', N, (int)P_X, Xc, (int)P_X, Sx, Ux, nx, Vtx, (int)P_X);
    
    // T = PCA scores (N x nx)
    double *T = Ux; // LAPACK already gave us the scores in 'S' mode
    for(int i=0; i<N; i++) for(int j=0; j<nx; j++) T[i*nx+j] *= Sx[j];

    // Z = T + Bias (N x nx+1)
    int nx_eff = nx + 1;
    double *Z = (double *)malloc((long)N * nx_eff * sizeof(double));
    expand_v4_double(T, Z, N, nx);

    // 3. Pixel-wise Magnitude-only Regression
    int nxp = xa_y / patchsize;
    int nyp = ya_y / patchsize;
    int n_patches = nxp * nyp;
    int target_dim_total = n_patches * ((ny == 0) ? (patchsize * patchsize) : ny);
    
    printf("Regression v4 for %d target components...\n", target_dim_total);
    double *B_re = (double *)malloc((long)target_dim_total * nx_eff * sizeof(double));
    double *B_im = (double *)malloc((long)target_dim_total * nx_eff * sizeof(double));
    
    // We compute patch by patch for cache efficiency
    #pragma omp parallel for collapse(2)
    for (int py = 0; py < nyp; py++) {
        for (int px = 0; px < nxp; px++) {
            int patch_idx = py * nxp + px;
            int patch_pixels = patchsize * patchsize;
            
            // Extract patch stack N x patch_pixels
            double *Y_patch = (double *)malloc((long)N * patch_pixels * sizeof(double));
            for (int i = 0; i < N; i++)
                for (int dy = 0; dy < patchsize; dy++)
                    for (int dx = 0; dx < patchsize; dx++)
                        Y_patch[i * patch_pixels + dy * patchsize + dx] = 
                            Y_raw[i * P_Y + (py * patchsize + dy) * xa_y + (px * patchsize + dx)];

            // For each pixel in patch
            for (int p = 0; p < patch_pixels; p++) {
                double *Y_pix = (double *)malloc(N * sizeof(double));
                for (int i = 0; i < N; i++) Y_pix[i] = Y_patch[i * patch_pixels + p];
                
                int comp_idx = patch_idx * patch_pixels + p;
                solve_v4_pixel(Z, Y_pix, &B_re[comp_idx * nx_eff], &B_im[comp_idx * nx_eff], 
                               N, nx_eff, max_iters, lr, reg_lambda);
                
                free(Y_pix);
            }
            free(Y_patch);
            if(patch_idx % 10 == 0) printf("  Processed patch %d/%d\n", patch_idx, n_patches);
        }
    }

    // 4. Save Outputs
    char path[1024];
    snprintf(path, 1024, "%s_B_re.fits", out_prefix);
    write_fits_2d(path, B_re, (long)nx_eff, (long)target_dim_total);
    snprintf(path, 1024, "%s_B_im.fits", out_prefix);
    write_fits_2d(path, B_im, (long)nx_eff, (long)target_dim_total);
    snprintf(path, 1024, "%s_PCx.fits", out_prefix);
    // Vtx is nx x P_X, but we want P_X x nx for consistency
    double *PCx = (double *)malloc((long)P_X * nx * sizeof(double));
    for(long p=0; p<P_X; p++) for(int k=0; k<nx; k++) PCx[p*nx+k] = Vtx[k*P_X+p];
    write_fits_2d(path, PCx, (long)nx, (long)P_X);
    
    snprintf(path, 1024, "%s_Xmean.fits", out_prefix);
    write_fits_2d(path, X_mean, P_X, 1);
    snprintf(path, 1024, "%s_Xstd.fits", out_prefix);
    write_fits_2d(path, X_std, P_X, 1);

    snprintf(path, 1024, "%s_v4_info.txt", out_prefix);
    FILE *fp = fopen(path, "w");
    fprintf(fp, "patchsize %d\nny_per_patch %d\nnxp %d\nnyp %d\nnx %d\n", 
            patchsize, (ny==0?patchsize*patchsize:ny), nxp, nyp, nx);
    fclose(fp);

    printf("v4 fit done. Model stored with prefix: %s\n", out_prefix);
    return 0;
}
