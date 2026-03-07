#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "common.h"

// Helper function to transpose a matrix (row-major src[rows x cols] -> dst[cols x rows])
void transpose_matrix_float(const float *src, float *dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

void transpose_matrix_double(const double *src, double *dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}


void print_help(const char *progname) {
    fprintf(stderr, "Usage: %s [options] <npca> <input.fits> <modes.fits> <coeffs.fits>\n", progname);
    fprintf(stderr, "\n");
    fprintf(stderr, "Performs Incremental Principal Component Analysis (SVD) on a 3D FITS cube (x, y, N).\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Arguments:\n");
    fprintf(stderr, "  <npca>          Number of principal components to compute.\n");
    fprintf(stderr, "  <input.fits>    Input 3D FITS cube (N samples).\n");
    fprintf(stderr, "  <modes.fits>    Output spatial modes (npca, y, x).\n");
    fprintf(stderr, "  <coeffs.fits>   Output temporal coefficients (N, npca).\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -ncpu <n>       Set number of CPU threads (OpenBLAS).\n");
    fprintf(stderr, "  -float          Use single precision (float) instead of double.\n");
    fprintf(stderr, "  -batchsize <n>  Number of samples to process in each batch.\n");
    fprintf(stderr, "                  Default: auto (largest batch fitting in ~1 GB).\n");
    fprintf(stderr, "\n");
}

// ============================================================================
// Memory-mapped FITS reader
// ============================================================================
// Opens a FITS file read-only and returns a memory-mapped pointer to the
// raw data array.  Also returns the dimensions (xa, ya, N) and the byte
// offset where pixel data starts.
//
// FITS stores data in big-endian.  On little-endian hosts the caller must
// byte-swap when copying into work buffers (the batch loop already copies
// for mean subtraction, so the swap is folded into that copy).
// ============================================================================

typedef struct {
    void   *map_base;    // mmap base pointer
    size_t  map_len;     // total mapped length
    size_t  data_offset; // byte offset of first pixel
    int     xa, ya;
    long    N, P;
    int     bitpix;      // FITS BITPIX (negative = IEEE float)
    int     need_swap;   // 1 if host is little-endian (FITS is big-endian)
} mmap_fits_t;

static int host_is_little_endian(void) {
    uint32_t x = 1;
    return *(uint8_t *)&x;
}

// Byte-swap helpers
static void bswap32_buf(void *dst, const void *src, size_t n) {
    const uint8_t *s = (const uint8_t *)src;
    uint8_t       *d = (uint8_t *)dst;
    for (size_t i = 0; i < n; i++, s += 4, d += 4) {
        d[0] = s[3]; d[1] = s[2]; d[2] = s[1]; d[3] = s[0];
    }
}
static void bswap64_buf(void *dst, const void *src, size_t n) {
    const uint8_t *s = (const uint8_t *)src;
    uint8_t       *d = (uint8_t *)dst;
    for (size_t i = 0; i < n; i++, s += 8, d += 8) {
        d[0] = s[7]; d[1] = s[6]; d[2] = s[5]; d[3] = s[4];
        d[4] = s[3]; d[5] = s[2]; d[6] = s[1]; d[7] = s[0];
    }
}

// Read a FITS header keyword value (very simple parser, enough for NAXIS etc.)
static long fits_header_long(const char *header, size_t hdr_len, const char *key) {
    size_t klen = strlen(key);
    for (size_t pos = 0; pos + 80 <= hdr_len; pos += 80) {
        if (strncmp(header + pos, key, klen) == 0 && header[pos + klen] == ' ') {
            // find '='
            const char *eq = memchr(header + pos, '=', 80);
            if (eq) return atol(eq + 1);
        }
    }
    return 0;
}

// Open + mmap a 3D FITS file.  Returns 0 on success.
static int mmap_fits_open(const char *filename, mmap_fits_t *mf) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }

    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return -1; }
    mf->map_len = (size_t)st.st_size;

    mf->map_base = mmap(NULL, mf->map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  // fd can be closed after mmap
    if (mf->map_base == MAP_FAILED) { perror("mmap"); return -1; }

    // Parse FITS header (each header block = 2880 bytes)
    const char *hdr = (const char *)mf->map_base;
    size_t hdr_end = 0;
    for (size_t pos = 0; pos < mf->map_len; pos += 80) {
        if (strncmp(hdr + pos, "END", 3) == 0 &&
            (hdr[pos+3] == ' ' || hdr[pos+3] == '\0')) {
            hdr_end = pos + 80;
            break;
        }
    }
    // Round up to next 2880-byte boundary
    mf->data_offset = ((hdr_end + 2879) / 2880) * 2880;

    int naxis = (int)fits_header_long(hdr, hdr_end, "NAXIS");
    if (naxis != 3) {
        fprintf(stderr, "Error: FITS file must be 3D (NAXIS=3), got %d\n", naxis);
        munmap(mf->map_base, mf->map_len);
        return -1;
    }
    mf->bitpix = (int)fits_header_long(hdr, hdr_end, "BITPIX");
    mf->xa = (int)fits_header_long(hdr, hdr_end, "NAXIS1");
    mf->ya = (int)fits_header_long(hdr, hdr_end, "NAXIS2");
    mf->N  = fits_header_long(hdr, hdr_end, "NAXIS3");
    mf->P  = (long)mf->xa * mf->ya;
    mf->need_swap = host_is_little_endian();

    return 0;
}

static void mmap_fits_close(mmap_fits_t *mf) {
    if (mf->map_base && mf->map_base != MAP_FAILED) {
        munmap(mf->map_base, mf->map_len);
    }
    mf->map_base = NULL;
}

// Copy a batch of frames from the mmap into a float work buffer, byte-swapping if needed.
static void mmap_read_batch_float(const mmap_fits_t *mf, long frame_offset, long nframes, float *dst) {
    long npix = nframes * mf->P;
    if (mf->bitpix == -32) {
        // Source is 32-bit IEEE float (big-endian in FITS)
        const uint8_t *src = (const uint8_t *)mf->map_base + mf->data_offset
                           + (size_t)frame_offset * mf->P * 4;
        if (mf->need_swap) {
            bswap32_buf(dst, src, (size_t)npix);
        } else {
            memcpy(dst, src, (size_t)npix * 4);
        }
    } else if (mf->bitpix == -64) {
        // Source is 64-bit double — read, swap, then down-convert
        const uint8_t *src = (const uint8_t *)mf->map_base + mf->data_offset
                           + (size_t)frame_offset * mf->P * 8;
        double *tmp = (double *)malloc((size_t)npix * sizeof(double));
        if (mf->need_swap) {
            bswap64_buf(tmp, src, (size_t)npix);
        } else {
            memcpy(tmp, src, (size_t)npix * 8);
        }
        for (long i = 0; i < npix; i++) dst[i] = (float)tmp[i];
        free(tmp);
    } else {
        // Unsupported BITPIX — fall through (should not happen for float mode)
        fprintf(stderr, "Warning: unsupported BITPIX %d for float mode, zeroing batch.\n", mf->bitpix);
        memset(dst, 0, (size_t)npix * sizeof(float));
    }
}

static void mmap_read_batch_double(const mmap_fits_t *mf, long frame_offset, long nframes, double *dst) {
    long npix = nframes * mf->P;
    if (mf->bitpix == -64) {
        const uint8_t *src = (const uint8_t *)mf->map_base + mf->data_offset
                           + (size_t)frame_offset * mf->P * 8;
        if (mf->need_swap) {
            bswap64_buf(dst, src, (size_t)npix);
        } else {
            memcpy(dst, src, (size_t)npix * 8);
        }
    } else if (mf->bitpix == -32) {
        // Source is 32-bit float — read, swap, then up-convert
        const uint8_t *src = (const uint8_t *)mf->map_base + mf->data_offset
                           + (size_t)frame_offset * mf->P * 4;
        float *tmp = (float *)malloc((size_t)npix * sizeof(float));
        if (mf->need_swap) {
            bswap32_buf(tmp, src, (size_t)npix);
        } else {
            memcpy(tmp, src, (size_t)npix * 4);
        }
        for (long i = 0; i < npix; i++) dst[i] = (double)tmp[i];
        free(tmp);
    } else if (mf->bitpix == 16) {
        // 16-bit signed integer
        const uint8_t *src = (const uint8_t *)mf->map_base + mf->data_offset
                           + (size_t)frame_offset * mf->P * 2;
        for (long i = 0; i < npix; i++) {
            int16_t val;
            if (mf->need_swap) {
                uint8_t tmp[2] = { src[i*2+1], src[i*2] };
                memcpy(&val, tmp, 2);
            } else {
                memcpy(&val, src + i*2, 2);
            }
            dst[i] = (double)val;
        }
    } else if (mf->bitpix == 32) {
        // 32-bit signed integer
        const uint8_t *src = (const uint8_t *)mf->map_base + mf->data_offset
                           + (size_t)frame_offset * mf->P * 4;
        for (long i = 0; i < npix; i++) {
            int32_t val;
            if (mf->need_swap) {
                uint8_t tmp[4] = { src[i*4+3], src[i*4+2], src[i*4+1], src[i*4] };
                memcpy(&val, tmp, 4);
            } else {
                memcpy(&val, src + i*4, 4);
            }
            dst[i] = (double)val;
        }
    } else {
        fprintf(stderr, "Warning: unsupported BITPIX %d for double mode, zeroing batch.\n", mf->bitpix);
        memset(dst, 0, (size_t)npix * sizeof(double));
    }
}


int main(int argc, char *argv[]) {
    int ncpu = 0;
    int use_float = 0;
    int arg_offset = 0;
    int batch_size = 0;          // 0 = auto
    int batch_size_user_set = 0;

    // Argument parsing
    int i = 1;
    while (i < argc) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-ncpu") == 0) {
            if (i + 1 >= argc) {
                print_args(argc, argv);
                fprintf(stderr, "Error: -ncpu requires an argument.\n");
                return 1;
            }
            ncpu = atoi(argv[i+1]);
            if (ncpu < 1) {
                print_args(argc, argv);
                fprintf(stderr, "Error: ncpu must be >= 1. Got %d.\n", ncpu);
                return 1;
            }
            i += 2;
            arg_offset += 2;
        } else if (strcmp(argv[i], "-float") == 0) {
            use_float = 1;
            i += 1;
            arg_offset += 1;
        } else if (strcmp(argv[i], "-batchsize") == 0) {
            if (i + 1 >= argc) {
                print_args(argc, argv);
                fprintf(stderr, "Error: -batchsize requires an argument.\n");
                return 1;
            }
            batch_size = atoi(argv[i+1]);
            batch_size_user_set = 1;
            if (batch_size < 1) {
                print_args(argc, argv);
                fprintf(stderr, "Error: batchsize must be >= 1. Got %d.\n", batch_size);
                return 1;
            }
            i += 2;
            arg_offset += 2;
        } else {
            break; // Positional argument found
        }
    }

    if (argc - arg_offset != 5) {
        print_args(argc, argv);
        fprintf(stderr, "Error: Missing required arguments.\n");
        print_help(argv[0]);
        return 1;
    }

    if (ncpu > 0) {
        openblas_set_num_threads(ncpu);
        printf("Using %d CPU cores.\n", ncpu);
    }
    if (use_float) {
        printf("Using Single Precision (float).\n");
    } else {
        printf("Using Double Precision (double).\n");
    }

    int npca = atoi(argv[1 + arg_offset]);
    const char *infile = argv[2 + arg_offset];
    const char *modes_file = argv[3 + arg_offset];
    const char *coeffs_file = argv[4 + arg_offset];

    if (npca < 1) {
        print_args(argc, argv);
        fprintf(stderr, "Error: npca must be >= 1. Got %d.\n", npca);
        return 1;
    }

    // ========================================================================
    // Memory-map the FITS file
    // ========================================================================
    mmap_fits_t mf;
    if (mmap_fits_open(infile, &mf) != 0) {
        fprintf(stderr, "Error: Could not memory-map %s\n", infile);
        return 1;
    }

    int xa = mf.xa;
    int ya = mf.ya;
    long N = mf.N;
    long P = mf.P;

    printf("  Input Dimensions: %ld samples x %ld pixels (%d x %d)\n", N, P, xa, ya);
    printf("  BITPIX: %d, byte-swap: %s\n", mf.bitpix, mf.need_swap ? "yes" : "no");

    if (npca > N) { npca = (int)N; }
    if (npca > P) { npca = (int)P; }

    // ========================================================================
    // Automatic batch-size selection
    // ========================================================================
    if (!batch_size_user_set) {
        long mem_limit = 1L * 1024 * 1024 * 1024; // 1 GB target
        long elem_size = use_float ? (long)sizeof(float) : (long)sizeof(double);
        long row_bytes = P * elem_size;
        if (row_bytes > 0) {
            batch_size = (int)(mem_limit / row_bytes);
        }
        if (batch_size < 2 * npca) batch_size = 2 * npca;
        if (batch_size > N) batch_size = (int)N;
        printf("  Auto batch size: %d  (targeting ~1 GB work buffer)\n", batch_size);
    }

    if (batch_size < npca) {
        int new_batch_size = 2 * npca;
        printf("Info: batch_size (%d) is smaller than npca (%d). Automatically adjusting batch_size to %d for stability.\n", batch_size, npca, new_batch_size);
        batch_size = new_batch_size;
    } else if (batch_size_user_set && batch_size < 2 * npca) {
        fprintf(stderr, "Warning: For stable results, it is recommended that batch_size is significantly larger than npca (e.g., >= 2 * npca). Current: batch_size=%d, npca=%d\n", batch_size, npca);
    }

    printf("Incremental PCA Configuration:\n");
    printf("  npca: %d\n", npca);
    printf("  Input: %s\n", infile);
    printf("  Modes Output: %s\n", modes_file);
    printf("  Coeffs Output: %s\n", coeffs_file);
    printf("  Batch Size: %d\n", batch_size);

    if (use_float) {
        // ====================================================================
        // IPCA implementation for float
        // ====================================================================

        printf("--- Pass 1: Computing Modes ---\n");

        float *V = (float*)malloc(npca * P * sizeof(float));
        float *S = (float*)malloc(npca * sizeof(float));
        long total_samples = 0;

        for (long frame_offset = 0; frame_offset < N; frame_offset += batch_size) {
            long b = (frame_offset + batch_size > N) ? (N - frame_offset) : batch_size;
            printf("Processing batch: frames %ld to %ld (size %ld)\n", frame_offset, frame_offset + b - 1, b);

            float *B = (float*)malloc(b * P * sizeof(float));
            mmap_read_batch_float(&mf, frame_offset, b, B);

            total_samples += b;

            if (total_samples == b) {
                // First batch: initialize via thin SVD of B
                int K = (b < P) ? (int)b : (int)P;
                if (npca > K) npca = K;

                float *Utmp = (float*)malloc(b*K*sizeof(float));
                float *Vt = (float*)malloc(K*P*sizeof(float));
                float *Stmp = (float*)malloc(K*sizeof(float));

                LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'S', (int)b, (int)P, B, (int)P, Stmp, Utmp, K, Vt, (int)P);

                for(int k=0; k<npca; k++) {
                    S[k] = Stmp[k];
                    memcpy(&V[k*P], &Vt[k*P], P*sizeof(float));
                }

                free(Utmp); free(Vt); free(Stmp);
            } else {
                // ============================================================
                // Incremental SVD update (Brand's algorithm)
                //
                // V is npca x P (row-major): current right singular vectors
                // S is npca:                  current singular values
                // B is b x P:                 new (centered) batch
                //
                // Steps:
                //  1. L = B * V^T       — projection onto current subspace (b x npca)
                //  2. H = B - L * V     — residual orthogonal to V          (b x P)
                //  3. QR(H^T) = Q * R   — thin QR of H^T (P x b)
                //     Q is P x b, R is b x b
                //  4. Build K-matrix (npca+b) x (npca+b):
                //        K = [ diag(S)   L^T ]
                //            [    0        R  ]
                //  5. SVD(K) -> truncate to top npca
                //  6. Update V = Vk(:,1:npca)^T * [V ; Q^T]  (combined rotation)
                // ============================================================

                int r = (int)b;    // residual has at most b directions
                int dim = npca + r;

                // Step 1: L = B * V^T  (b x npca)
                float *L = (float*)malloc(b*npca*sizeof(float));
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            (int)b, npca, (int)P, 1.0f, B, (int)P, V, (int)P, 0.0f, L, npca);

                // Step 2: H = B - L * V  (b x P)
                float *H = (float*)malloc(b*P*sizeof(float));
                memcpy(H, B, b*P*sizeof(float));
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            (int)b, (int)P, npca, -1.0f, L, npca, V, (int)P, 1.0f, H, (int)P);

                // Step 3: QR factorization of H^T (P x b)
                float *Ht = (float*)malloc(P*b*sizeof(float));
                transpose_matrix_float(H, Ht, (int)b, (int)P);

                float *tau = (float*)malloc(r*sizeof(float));
                LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, (int)P, r, Ht, r, tau);

                // Extract R (upper triangular, r x r)
                float *Rmat = (float*)calloc(r*r, sizeof(float));
                for(int ii=0; ii<r; ii++) {
                    for(int jj=ii; jj<r; jj++) {
                        Rmat[ii*r+jj] = Ht[ii*r+jj];
                    }
                }

                // Generate Q in-place in Ht (P x r, stored row-major with lda=r)
                LAPACKE_sorgqr(LAPACK_ROW_MAJOR, (int)P, r, r, Ht, r, tau);

                // Step 4: Build K-matrix (dim x dim)
                // K = [ diag(S)   L^T ]
                //     [    0       R  ]
                float *Kmat = (float*)calloc(dim*dim, sizeof(float));
                // Upper-left: diag(S)
                for(int ii=0; ii<npca; ii++) Kmat[ii*dim+ii] = S[ii];
                // Upper-right: L^T (npca x b) — L is b x npca, so L^T[i][j] = L[j*npca+i]
                for(int ii=0; ii<npca; ii++) {
                    for(int jj=0; jj<r; jj++) {
                        Kmat[ii*dim + npca + jj] = L[jj*npca+ii];
                    }
                }
                // Lower-right: R (r x r)
                for(int ii=0; ii<r; ii++) {
                    for(int jj=0; jj<r; jj++) {
                        Kmat[(npca+ii)*dim + (npca+jj)] = Rmat[ii*r+jj];
                    }
                }

                // Step 5: SVD of K (dim x dim)
                float *S2 = (float*)malloc(dim*sizeof(float));
                float *U2 = (float*)malloc(dim*dim*sizeof(float));
                float *V2t = (float*)malloc(dim*dim*sizeof(float));
                LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'A', dim, dim, Kmat, dim, S2, U2, dim, V2t, dim);

                // Step 6: Update V
                // New V^T(P x npca) = [V^T | Q] (P x dim) * U2(:, 0:npca) (dim x npca)
                //
                // Build V_and_Q row-by-row in row-major order:
                //   V_and_Q[p, k]       = V^T[p, k] = V[k*P + p]   for k in [0, npca)
                //   V_and_Q[p, npca+j]  = Q[p, j]   = Ht[p*r + j]  for j in [0, r)
                float *V_and_Q = (float*)malloc(P * dim * sizeof(float));
                for (long p = 0; p < P; p++) {
                    for (int k = 0; k < npca; k++)
                        V_and_Q[p * dim + k] = V[k * P + p];
                    for (int jj = 0; jj < r; jj++)
                        V_and_Q[p * dim + npca + jj] = Ht[p * r + jj];
                }

                // Vnew_t (P x npca) = V_and_Q (P x dim) * U2(:, 0:npca) (dim x npca)
                float *Vnew_t = (float*)malloc(P * npca * sizeof(float));
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            (int)P, npca, dim, 1.0f, V_and_Q, dim, U2, dim, 0.0f, Vnew_t, npca);

                // Transpose Vnew_t (P x npca) -> V (npca x P)
                transpose_matrix_float(Vnew_t, V, (int)P, npca);
                for(int k=0; k<npca; k++) S[k] = S2[k];

                free(L); free(H); free(Ht); free(tau); free(Rmat); free(Kmat);
                free(S2); free(U2); free(V2t);
                free(V_and_Q); free(Vnew_t);
            }
            free(B);
        }

        printf("--- Pass 1 Complete ---\n");

        // Write Eigenvalues
        FILE *feig = fopen("ipca.eigenvalues.txt", "w");
        if (feig) {
            for(int k=0; k<npca; k++) {
                fprintf(feig, "%.6g\n", S[k]*S[k]);
            }
            fclose(feig);
            printf("Wrote ipca.eigenvalues.txt\n");
        } else {
            fprintf(stderr, "Warning: Could not write ipca.eigenvalues.txt\n");
        }

        printf("Writing Modes to %s...\n", modes_file);
        write_fits_3d_float(modes_file, V, xa, ya, npca);

        printf("--- Pass 2: Computing Coefficients ---\n");

        int status = 0;
        fitsfile *coeffs_fptr;
        long coeffs_naxes[2] = {npca, N};
        remove(coeffs_file);
        fits_create_file(&coeffs_fptr, coeffs_file, &status);
        CHECK_STATUS(status);
        fits_create_img(coeffs_fptr, FLOAT_IMG, 2, coeffs_naxes, &status);
        CHECK_STATUS(status);

        for (long frame_offset = 0; frame_offset < N; frame_offset += batch_size) {
            long current_batch_size = (frame_offset + batch_size > N) ? (N - frame_offset) : batch_size;
            float *X_batch = (float*)malloc(current_batch_size * P * sizeof(float));
            mmap_read_batch_float(&mf, frame_offset, current_batch_size, X_batch);



            float *Coeffs_batch = (float*)malloc(current_batch_size * npca * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)current_batch_size, npca, (int)P, 1.0f, X_batch, (int)P, V, (int)P, 0.0f, Coeffs_batch, npca);

            long fpixel_write[2] = {1, frame_offset + 1};
            fits_write_pix(coeffs_fptr, TFLOAT, fpixel_write, current_batch_size * npca, Coeffs_batch, &status);
            CHECK_STATUS(status);

            free(X_batch);
            free(Coeffs_batch);
        }

        fits_close_file(coeffs_fptr, &status);
        CHECK_STATUS(status);
        printf("--- Pass 2 Complete ---\n");

        free(V); free(S);
    } else {
        // ====================================================================
        // IPCA implementation for double
        // ====================================================================

        printf("--- Pass 1: Computing Modes ---\n");

        double *V = (double*)malloc(npca * P * sizeof(double));
        double *S = (double*)malloc(npca * sizeof(double));
        long total_samples = 0;

        for (long frame_offset = 0; frame_offset < N; frame_offset += batch_size) {
            long b = (frame_offset + batch_size > N) ? (N - frame_offset) : batch_size;
            printf("Processing batch: frames %ld to %ld (size %ld)\n", frame_offset, frame_offset + b - 1, b);

            double *B = (double*)malloc(b * P * sizeof(double));
            mmap_read_batch_double(&mf, frame_offset, b, B);

            total_samples += b;

            if (total_samples == b) {
                // First batch: initialize via thin SVD of B
                int K = (b < P) ? (int)b : (int)P;
                if (npca > K) npca = K;

                double *Utmp = (double*)malloc(b*K*sizeof(double));
                double *Vt = (double*)malloc(K*P*sizeof(double));
                double *Stmp = (double*)malloc(K*sizeof(double));

                LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', (int)b, (int)P, B, (int)P, Stmp, Utmp, K, Vt, (int)P);

                for(int k=0; k<npca; k++) {
                    S[k] = Stmp[k];
                    memcpy(&V[k*P], &Vt[k*P], P*sizeof(double));
                }

                free(Utmp); free(Vt); free(Stmp);
            } else {
                // ============================================================
                // Incremental SVD update (Brand's algorithm) — double path
                // ============================================================

                int r = (int)b;
                int dim = npca + r;

                // Step 1: L = B * V^T  (b x npca)
                double *L = (double*)malloc(b*npca*sizeof(double));
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            (int)b, npca, (int)P, 1.0, B, (int)P, V, (int)P, 0.0, L, npca);

                // Step 2: H = B - L * V  (b x P)
                double *H = (double*)malloc(b*P*sizeof(double));
                memcpy(H, B, b*P*sizeof(double));
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            (int)b, (int)P, npca, -1.0, L, npca, V, (int)P, 1.0, H, (int)P);

                // Step 3: QR factorization of H^T (P x b)
                double *Ht = (double*)malloc(P*b*sizeof(double));
                transpose_matrix_double(H, Ht, (int)b, (int)P);

                double *tau = (double*)malloc(r*sizeof(double));
                LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, (int)P, r, Ht, r, tau);

                // Extract R (upper triangular, r x r)
                double *Rmat = (double*)calloc(r*r, sizeof(double));
                for(int ii=0; ii<r; ii++) {
                    for(int jj=ii; jj<r; jj++) {
                        Rmat[ii*r+jj] = Ht[ii*r+jj];
                    }
                }

                // Generate Q in-place in Ht (P x r, row-major with lda=r)
                LAPACKE_dorgqr(LAPACK_ROW_MAJOR, (int)P, r, r, Ht, r, tau);

                // Step 4: Build K-matrix (dim x dim)
                double *Kmat = (double*)calloc(dim*dim, sizeof(double));
                for(int ii=0; ii<npca; ii++) Kmat[ii*dim+ii] = S[ii];
                for(int ii=0; ii<npca; ii++) {
                    for(int jj=0; jj<r; jj++) {
                        Kmat[ii*dim + npca + jj] = L[jj*npca+ii];
                    }
                }
                for(int ii=0; ii<r; ii++) {
                    for(int jj=0; jj<r; jj++) {
                        Kmat[(npca+ii)*dim + (npca+jj)] = Rmat[ii*r+jj];
                    }
                }

                // Step 5: SVD of K (dim x dim)
                double *S2 = (double*)malloc(dim*sizeof(double));
                double *U2 = (double*)malloc(dim*dim*sizeof(double));
                double *V2t = (double*)malloc(dim*dim*sizeof(double));
                LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', dim, dim, Kmat, dim, S2, U2, dim, V2t, dim);

                // Step 6: Update V
                double *V_and_Q = (double*)malloc(P * dim * sizeof(double));
                for (long p = 0; p < P; p++) {
                    for (int k = 0; k < npca; k++)
                        V_and_Q[p * dim + k] = V[k * P + p];
                    for (int jj = 0; jj < r; jj++)
                        V_and_Q[p * dim + npca + jj] = Ht[p * r + jj];
                }

                double *Vnew_t = (double*)malloc(P * npca * sizeof(double));
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            (int)P, npca, dim, 1.0, V_and_Q, dim, U2, dim, 0.0, Vnew_t, npca);

                transpose_matrix_double(Vnew_t, V, (int)P, npca);
                for(int k=0; k<npca; k++) S[k] = S2[k];

                free(L); free(H); free(Ht); free(tau); free(Rmat); free(Kmat);
                free(S2); free(U2); free(V2t);
                free(V_and_Q); free(Vnew_t);
            }
            free(B);
        }

        printf("--- Pass 1 Complete ---\n");

        // Write Eigenvalues
        FILE *feig = fopen("ipca.eigenvalues.txt", "w");
        if (feig) {
            for(int k=0; k<npca; k++) {
                fprintf(feig, "%.15g\n", S[k]*S[k]);
            }
            fclose(feig);
            printf("Wrote ipca.eigenvalues.txt\n");
        } else {
            fprintf(stderr, "Warning: Could not write ipca.eigenvalues.txt\n");
        }

        printf("Writing Modes to %s...\n", modes_file);
        write_fits_3d(modes_file, V, xa, ya, npca);

        printf("--- Pass 2: Computing Coefficients ---\n");

        int status = 0;
        fitsfile *coeffs_fptr;
        long coeffs_naxes[2] = {npca, N};
        remove(coeffs_file);
        fits_create_file(&coeffs_fptr, coeffs_file, &status);
        CHECK_STATUS(status);
        fits_create_img(coeffs_fptr, DOUBLE_IMG, 2, coeffs_naxes, &status);
        CHECK_STATUS(status);

        for (long frame_offset = 0; frame_offset < N; frame_offset += batch_size) {
            long current_batch_size = (frame_offset + batch_size > N) ? (N - frame_offset) : batch_size;
            double *X_batch = (double*)malloc(current_batch_size * P * sizeof(double));
            mmap_read_batch_double(&mf, frame_offset, current_batch_size, X_batch);



            double *Coeffs_batch = (double*)malloc(current_batch_size * npca * sizeof(double));
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)current_batch_size, npca, (int)P, 1.0, X_batch, (int)P, V, (int)P, 0.0, Coeffs_batch, npca);

            long fpixel_write[2] = {1, frame_offset + 1};
            fits_write_pix(coeffs_fptr, TDOUBLE, fpixel_write, current_batch_size * npca, Coeffs_batch, &status);
            CHECK_STATUS(status);

            free(X_batch);
            free(Coeffs_batch);
        }

        fits_close_file(coeffs_fptr, &status);
        CHECK_STATUS(status);
        printf("--- Pass 2 Complete ---\n");

        free(V); free(S);
    }

    mmap_fits_close(&mf);

    printf("Done.\n");

    return 0;
}
