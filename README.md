# milk-streamtelemetry-lintools

## Description

This suite of tools performs Canonical Correlation Analysis (CCA) and Principal Component Analysis (PCA) on series of images (telemetry streams).
The input typically consists of 3D FITS cubes (dimensions x, y, N), representing a series of N images.

The suite includes the following programs:
*   `milk-streamtelemetry-pca`: Performs PCA (SVD) on a single dataset.
*   `milk-streamtelemetry-cca`: Performs CCA between two datasets.
*   `milk-streamtelemetry-recon`: Reconstructs datasets from modes and coefficients.
*   `milk-streamtelemetry-pcamerge`: Merges multiple PCA results into a global basis.
*   `milk-streamtelemetry-cubeshuffle`: Randomly shuffles the slices of a 3D FITS cube.
*   `milk-streamtelemetry-wfs2psf-fit`: Fits a WFS→PSF quadratic mapping model from telemetry and image cubes.
*   `milk-streamtelemetry-wfs2psf-predict`: Applies a fitted WFS→PSF model to generate predicted PSF cubes from telemetry.
*   `milk-streamtelemetry-wfs2psf-update`: Incrementally updates an existing WFS→PSF model with new data.

Dependencies: CFITSIO, OpenBLAS, LAPACKE.

## Compilation

Coded in C, using gcc and cmake for compilation.
```bash
mkdir build
cd build
cmake ..
make
```

## Usage

All programs support CFITSIO's extended filename syntax, allowing you to read subsets (crops) of images.
Example: `input.fits[1:50,1:50,1:1000]`.

### 1. Principal Component Analysis (PCA)

Performs SVD on the dataset to extract spatial modes and temporal coefficients.
Note: The data is **not** de-averaged (centered) before analysis. The first mode captures the average signal and is forced to have a positive average coefficient.

```bash
milk-streamtelemetry-pca [options] <npca> <input.fits> <modes.fits> <coeffs.fits>
```

**Arguments:**
*   `<npca>`: Number of principal components to compute.
*   `<input.fits>`: Input 3D FITS cube (x, y, N).
*   `<modes.fits>`: Output 3D FITS file containing spatial modes (npca, y, x).
*   `<coeffs.fits>`: Output 2D FITS file containing coefficients (N, npca).

**Options:**
*   `-ncpu <n>`: Number of CPU cores to use for linear algebra operations.
*   `-float`: Perform computations in single precision (float) instead of double. Reduces memory usage.
*   `-mask <file>`: Apply a spatial mask (FITS file, active where > 0.5). SVD is performed only on active pixels.
*   `-automask <arg>`: Automatically compute and save a mask (`pca.automask.fits`) based on variance.
    *   `n<val>`: Keep top `val` pixels (e.g., `n1000`).
    *   `f<val>`: Keep top `val` fraction of pixels (e.g., `f0.5`).

**Auxiliary Output:**
*   `pca.eigenvalues.txt`: Text file containing the squared singular values (eigenvalues of covariance).

### 2. Canonical Correlation Analysis (CCA)

Performs CCA between two datasets (A and B). This program operates in two modes depending on the input:

#### Mode A: Raw Image Cubes (Direct or Internal PCA)
Takes two image cubes, performs CCA (optionally reducing dimension via internal PCA first), and outputs spatial canonical vectors.

```bash
milk-streamtelemetry-cca [options] <nvec> <A.fits> <B.fits>
```
*   `<nvec>`: Number of canonical vectors to compute.
*   `<A.fits>`, `<B.fits>`: Input 3D image cubes.
*   **Output**: `ccaA.fits` and `ccaB.fits` (3D cubes of spatial canonical vectors).

#### Mode B: Modular Workflow (CCA on Coefficients)
Takes two 2D coefficient matrices (produced by `milk-streamtelemetry-pca`), performs CCA, and outputs the resulting canonical weights as 2D matrices.

```bash
milk-streamtelemetry-cca [options] <nvec> <coeffsA.fits> <coeffsB.fits>
```
*   `<coeffsA.fits>`, `<coeffsB.fits>`: Input 2D coefficient matrices (from PCA).
*   **Output**: `ccaA.fits` and `ccaB.fits` (2D matrices of canonical weights).

**Options:**
*   `-npca <n>`: (Mode A only) Perform PCA on inputs first keeping `n` modes, then run CCA on coefficients.
*   `-ncpu <n>`: Number of CPU cores to use.
*   `-float`: Perform computations in single precision.
*   `-shift <n>`: Shift the second series (B) by `n` time steps. Truncates non-overlapping samples.

### 3. Reconstruction

Reconstructs a dataset (or spatial maps) by multiplying a coefficient matrix with a mode matrix.

```bash
milk-streamtelemetry-recon <modes.fits> <coeffs.fits> <output.fits>
```
*   `<modes.fits>`: Spatial modes (e.g., from PCA output).
*   `<coeffs.fits>`: Coefficients (e.g., from PCA output or CCA Mode B output).
*   `<output.fits>`: Reconstructed 3D cube.

### 4. PCA Merge

Merges multiple partial PCA results (spatial modes and temporal coefficients) into a single global PCA basis. Useful for processing large datasets in chunks.

```bash
milk-streamtelemetry-pcamerge [-ncpu <n>] [-nmodes <n>] <input_list.txt> <output_modes.fits> <output_coeffs.fits>
```
*   `<input_list.txt>`: ASCII file listing input file pairs (one pair per line: `modes.fits coeffs.fits`).
*   `<output_modes.fits>`: Output global modes.
*   `<output_coeffs.fits>`: Output global coefficients.
*   `-nmodes <n>`: Number of global modes to retain.

### 5. Cube Shuffle

Randomly shuffles the slices (time axis) of a 3D FITS cube. Used for testing and destroying temporal correlations.

```bash
milk-streamtelemetry-cubeshuffle <input.fits> <output.fits>
```

## Examples

### Standard CCA on full images with masking
```bash
milk-streamtelemetry-pca -automask f0.5 100 A.fits modesA.fits coeffsA.fits
milk-streamtelemetry-pca -mask pca.automask.fits 100 B.fits modesB.fits coeffsB.fits
milk-streamtelemetry-cca 10 coeffsA.fits coeffsB.fits
milk-streamtelemetry-recon modesA.fits ccaA.fits spatial_canonical_A.fits
```

### Time-shifted CCA
```bash
milk-streamtelemetry-cca -shift 5 10 A.fits B.fits
```

---

## WFS→PSF Model (wfs2psf)

These tools fit a quadratic mapping model from wavefront sensor telemetry (`cube_w.fits`) to focal-plane images (`cube_i.fits`), and then apply the model to generate PSF predictions.

### Fit

Trains the model on a dataset folder.

```bash
milk-streamtelemetry-wfs2psf-fit cube_w.fits cube_i.fits <out_prefix> -nx <n_components>
```

**Minimum example:**
```bash
milk-streamtelemetry-wfs2psf-fit cube_wsci.fits cube_isci.fits model1 -nx 50
```

This produces `model1_B_latent.fits`, `model1_PCx.fits`, `model1_Xmean.fits`, `model1_Ymean.fits` (and patch basis files), which are all needed by the predict step.

**Key options:**
*   `-nx <int>`: Number of WFS PCA components (default: 10). The most impactful parameter.
*   `-ny <int>`: Number of PSF PCA components per patch (default: 5; set to 0 for raw pixels).
*   `-patchsize <int>`: Spatial patch size for local PSF decomposition (default: 16).
*   `-reg <float>`: Ridge regularization factor (default: 1.0).
*   `-trainsize <int>`: Limit training to the first N frames.
*   `-float`: Use single-precision arithmetic (faster, less memory).

### Predict

Applies a previously fitted model to a WFS telemetry cube to predict PSF frames.

```bash
milk-streamtelemetry-wfs2psf-predict cube_w.fits <out_prefix> <predicted_psf.fits>
```

**Minimum example:**
```bash
milk-streamtelemetry-wfs2psf-predict cube_wsci.fits model1 cube_ipred.fits
```

The `<out_prefix>` must match the prefix used during `wfs2psf-fit` so that the tool can locate all model files.
