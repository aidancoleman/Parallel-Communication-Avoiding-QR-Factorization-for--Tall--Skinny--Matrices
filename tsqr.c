#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <lapacke.h>
#include <cblas.h>
#include <string.h>

// Prototype for merging the Q matrices function.
// Here, blockRows is the number of rows in the Q block (local Q),
// and ncol is the number of columns (= dimy).
void apply_merge_Q_update(double* Q_sub, int blockRows, double* Q_merge, int ncol, int is_top);

// Frobenius norm difference between matrices A and B. Frobenius norm is the square root of the sum of all squared entries of matrix
double frobenius_norm_diff(double* A, double* B, int rows, int cols) {
    double sum = 0.0;
    int total = rows * cols;
    for (int i = 0; i < total; i++) {
        double diff = A[i] - B[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Frobenius norm of matrix A.
double frobenius_norm(double* A, int rows, int cols) {
    double sum = 0.0;
    int total = rows * cols;
    for (int i = 0; i < total; i++) {
        sum += A[i] * A[i];
    }
    return sqrt(sum);
}

/* Print a matrix of size rows x cols. */
void print_matrix(const char* name, double *mat, int rows, int cols) {
    printf("\n%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            printf("%10.4f ", mat[i*cols+j]);
        }
        printf("\n");
    }
}

/*
  Economy QR factorization on an m x n matrix (m >= n).
  On output, A contains Q (m x n) and the upper triangular factor R
  is stored in the upper triangle of the first n rows.
  Here, we set k = n.
*/
int qr_factor(double *A, int m, int n) {
    int lda = n;  // for economy size Q, storage is m x n
    int k = n;    // number of reflectors = n
    double *tau = (double*) malloc(k * sizeof(double));
    if (!tau) {
        printf("Failed to allocate tau.\n");
        return 1;
    }
    int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, A, lda, tau);
    if (info != 0) {
        printf("dgeqrf failed with info=%d.\n", info);
        free(tau);
        return 1;
    }
    // Generate the economy Q: A becomes Q (m x n) with leading dimension n.
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, n, k, A, lda, tau);
    if (info != 0) {
        printf("dorgqr failed with info=%d.\n", info);
        free(tau);
        return 1;
    }
    free(tau);
    return 0;
}

/*
  Merge two R blocks (each of size n x n) into a merged QR factorization.
  The two R blocks are stacked vertically to form a (2*n x n) matrix.
  We perform an economy QR on that matrix so that:
     Q_merged is (2*n x n) and R_merged is (n x n).
*/
int merge_R(double *R_top, double *R_bot, int n, double *Q_merged, double *R_merged) {
    int m = 2 * n;  // number of rows in the merged matrix
    int lda = n;    // economy storage for an m x n matrix
    int k = n;      // number of reflectors = n
    double *bigR = (double*) malloc(m * n * sizeof(double));
    if (!bigR) {
        printf("merge_R: allocation failed for bigR.\n");
        return 1;
    }
    // Copy R_top into the first n rows
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            bigR[i*n+j] = R_top[i*n+j];
        }
    }
    // Copy R_bot into the next n rows
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            bigR[(n+i)*n+j] = R_bot[i*n+j];
        }
    }
    double *tau = (double*) malloc(k * sizeof(double));
    if (!tau) {
        free(bigR);
        printf("merge_R: allocation failed for tau.\n");
        return 1;
    }
    int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, bigR, lda, tau);
    if (info != 0) {
        printf("merge_R: dgeqrf failed with info=%d.\n", info);
        free(tau);
        free(bigR);
        return 1;
    }
    // Extract R_merged from the upper triangle of bigR (first n rows)
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            R_merged[i*n+j] = (i <= j) ? bigR[i*n+j] : 0.0;
        }
    }
    // Generate economy Q_merged (m x n) from the reflectors.
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, n, k, bigR, lda, tau);
    if (info != 0) {
        printf("merge_R: dorgqr failed with info=%d.\n", info);
        free(tau);
        free(bigR);
        return 1;
    }
    // Copy Q_merged out
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            Q_merged[i*n+j] = bigR[i*n+j];
        }
    }
    free(tau);
    free(bigR);
    return 0;
}

/*
  Update a local Q block by multiplying on the right by a merge factor.
  Q_sub is of size (blockRows x ncol) and Q_merge is of size (2*ncol x ncol).
  If is_top is 1, we use the top ncol rows of Q_merge; otherwise, the bottom ncol rows.
  The update is: Q_sub := Q_sub * (selected ncol x ncol block of Q_merge).
*/
void apply_merge_Q_update(double* Q_sub, int blockRows, double* Q_merge, int ncol, int is_top) {
    int r_offset = is_top ? 0 : ncol;
    double* merge_sub = (double*) malloc(ncol * ncol * sizeof(double));
    if (!merge_sub) {
        printf("apply_merge_Q_update: allocation failed for merge_sub.\n");
        return;
    }
    // Extract the appropriate ncol x ncol submatrix from Q_merge.
    for (int i = 0; i < ncol; i++){
        for (int j = 0; j < ncol; j++){
            merge_sub[i*ncol+j] = Q_merge[(r_offset + i)*ncol + j];
        }
    }
    double* result = (double*) calloc(blockRows * ncol, sizeof(double));
    if (!result) {
        printf("apply_merge_Q_update: allocation failed for result.\n");
        free(merge_sub);
        return;
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                blockRows, ncol, ncol,
                1.0, Q_sub, ncol,
                merge_sub, ncol,
                0.0, result, ncol);
    for (int i = 0; i < blockRows * ncol; i++){
        Q_sub[i] = result[i];
    }
    free(merge_sub);
    free(result);
}

/*
  Compute and print the residual of a local block's QR factorization.
  A_local is the original block (blockRows x ncol),
  Q_local is (blockRows x ncol), R_local is (ncol x ncol).
*/
void compute_local_residual(double* A_local, double* Q_local, double* R_local, int blockRows, int ncol) {
    double* A_approx = (double*) malloc(blockRows * ncol * sizeof(double));
    if (!A_approx) {
        printf("Allocation failed for A_approx in compute_local_residual.\n");
        return;
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                blockRows, ncol, ncol,
                1.0, Q_local, ncol,
                R_local, ncol,
                0.0, A_approx, ncol);
    double res = frobenius_norm_diff(A_local, A_approx, blockRows, ncol);
    printf("Local residual (||A_local - Q_local*R_local||_F): %e\n", res);
    free(A_approx);
}

/*
  Check orthogonality of a matrix Q (of size m x n) by computing the Frobenius norm of (Q^T Q - I). Ideally this is very small << 1.
*/
void check_orthogonality(double* Q, int m, int n) {
    double* QtQ = (double*) calloc(n * n, sizeof(double));
    if (!QtQ) {
        printf("Allocation failed for QtQ in check_orthogonality.\n");
        return;
    }
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n, n, m,
                1.0, Q, n,
                Q, n,
                0.0, QtQ, n);
    for (int i = 0; i < n; i++) {
        QtQ[i*n+i] -= 1.0;
    }
    double ortho_error = frobenius_norm(QtQ, n, n);
    printf("Orthogonality error (||Q^T*Q - I||_F): %e\n", ortho_error);
    free(QtQ);
}

/*
  Main TSQR demonstration using economy QR.
  The overall matrix A is of size (dimx x dimy) where dimx is divisible by 4.
  Each local block is of size (block_size x dimy) and we compute an economy QR:
    Q_local: (block_size x dimy) and R_local: (dimy x dimy).
  Then we merge the R factors from two blocks (each dimy x dimy) by stacking them
  to form a (2*dimy x dimy) matrix and performing an economy QR.
  Finally, the final Q is assembled from updated local Q’s.
  Also measure the time taken for the TSQR procedure.
 */
int main() {
    int num_threads = 4; //default

    omp_set_num_threads(num_threads);
    
    int dimx, dimy;
    printf("Enter matrix dimensions (rows columns): ");
    scanf("%d %d", &dimx, &dimy);

    if (dimx % 4 != 0) {
        printf("Row dimension must be divisible by 4!\n");
        return 1;
    }
    int block_size = dimx / 4;
    // Dynamically allocate memory for the full matrix A (dimx x dimy)
    double *A = (double*) malloc(dimx * dimy * sizeof(double));
    if (!A) {
        printf("Allocation of A failed.\n"); // error check
        return 1;
    }
    srand(time(NULL));
    for (int i = 0; i < dimx; i++){
        for (int j = 0; j < dimy; j++){
            A[i*dimy+j] = (double) rand() / RAND_MAX;
        }
    }
    print_matrix("Randomly Generated Matrix A:", A, dimx, dimy);

    // Start timer for TSQR procedure
    double tsqr_start = omp_get_wtime();

    // --- Local QR for each of 4 blocks ---
    // Each block is block_size x dimy.
    // Allocate local Q_i (block_size x dimy) and local R_i (dimy x dimy).
    double *Q0 = (double*) malloc(block_size * dimy * sizeof(double));
    double *Q1 = (double*) malloc(block_size * dimy * sizeof(double));
    double *Q2 = (double*) malloc(block_size * dimy * sizeof(double));
    double *Q3 = (double*) malloc(block_size * dimy * sizeof(double));
    double *R0 = (double*) malloc(dimy * dimy * sizeof(double));
    double *R1 = (double*) malloc(dimy * dimy * sizeof(double));
    double *R2 = (double*) malloc(dimy * dimy * sizeof(double));
    double *R3 = (double*) malloc(dimy * dimy * sizeof(double));
    if (!Q0 || !Q1 || !Q2 || !Q3 || !R0 || !R1 || !R2 || !R3) {
        printf("Allocation failed for local Q or R.\n");
        return 1;
    }
    #pragma omp parallel for
    for (int b = 0; b < 4; b++){
        double *blockA = A + b * block_size * dimy; // pointer to block b (block_size x dimy)
        double *temp = (double*) malloc(block_size * dimy * sizeof(double));
        if (!temp) {
            printf("Allocation failed for temp in block %d.\n", b);
            continue;
        }
        for (int i = 0; i < block_size * dimy; i++){
            temp[i] = blockA[i];
        }
        int info = qr_factor(temp, block_size, dimy);
        if (info != 0){
            printf("Local QR failed on block %d.\n", b);
        }
        double *myQ, *myR;
        if (b == 0) { myQ = Q0; myR = R0; }
        else if (b == 1) { myQ = Q1; myR = R1; }
        else if (b == 2) { myQ = Q2; myR = R2; }
        else { myQ = Q3; myR = R3; }
        // Copy the computed Q (block_size x dimy)
        for (int i = 0; i < block_size; i++){
            for (int j = 0; j < dimy; j++){
                myQ[i*dimy+j] = temp[i*dimy+j];
            }
        }
        // The R factor is the upper triangular part of the first dimy rows of temp.
        for (int i = 0; i < dimy; i++){
            for (int j = 0; j < dimy; j++){
                myR[i*dimy+j] = (i <= j) ? temp[i*dimy+j] : 0.0;
            }
        }
        // Compute and print the local residual for this block. Prevent race conditions.
        #pragma omp critical
        {
            printf("\nBlock %d:\n", b);
            compute_local_residual(blockA, myQ, myR, block_size, dimy);
        }
        free(temp);
    }
    // --- End Local QR ---

    // --- Merge pairs of R's ---
    // Merge (R0, R1) -> Q01 (2*dimy x dimy), R01 (dimy x dimy)
    // Merge (R2, R3) -> Q23 (2*dimy x dimy), R23 (dimy x dimy)
    double *Q01 = (double*) malloc(2 * dimy * dimy * sizeof(double));
    double *R01 = (double*) malloc(dimy * dimy * sizeof(double));
    double *Q23 = (double*) malloc(2 * dimy * dimy * sizeof(double));
    double *R23 = (double*) malloc(dimy * dimy * sizeof(double));
    if (!Q01 || !R01 || !Q23 || !R23) {
        printf("Allocation failed for merged Q or R.\n");
        return 1;
    }
    merge_R(R0, R1, dimy, Q01, R01);
    merge_R(R2, R3, dimy, Q23, R23);
    // Update the local Q's using the merged Q factors.
    // Q0 and Q1 (each block_size x dimy) are updated:
    // Q0 := Q0 * (top half of Q01), Q1 := Q1 * (bottom half of Q01).
    apply_merge_Q_update(Q0, block_size, Q01, dimy, 1);
    apply_merge_Q_update(Q1, block_size, Q01, dimy, 0);
    apply_merge_Q_update(Q2, block_size, Q23, dimy, 1);
    apply_merge_Q_update(Q3, block_size, Q23, dimy, 0);
    // --- End Merge Pairs ---

    // --- Final Merge ---
    // Merge (R01, R23) to get final R.
    double *Q02 = (double*) malloc(2 * dimy * dimy * sizeof(double)); // Q02: (2*dimy x dimy)
    double *R02 = (double*) malloc(dimy * dimy * sizeof(double));       // R02: (dimy x dimy)
    if (!Q02 || !R02) {
        printf("Allocation failed for final merge.\n");
        return 1;
    }
    merge_R(R01, R23, dimy, Q02, R02);
    // update the global Q by merging the two pairs.
    // Stack Q0 and Q1 vertically to form Q01_big ( (2*block_size) x dimy )
    // and similarly stack Q2 and Q3 into Q23_big.
    double *Q01_big = (double*) malloc(2 * block_size * dimy * sizeof(double));
    double *Q23_big = (double*) malloc(2 * block_size * dimy * sizeof(double));
    if (!Q01_big || !Q23_big) {
        printf("Allocation failed for Q_big arrays.\n");
        return 1;
    }
    for (int i = 0; i < block_size; i++){
        for (int j = 0; j < dimy; j++){
            Q01_big[i*dimy+j] = Q0[i*dimy+j];
        }
    }
    for (int i = 0; i < block_size; i++){
        for (int j = 0; j < dimy; j++){
            Q01_big[(block_size+i)*dimy+j] = Q1[i*dimy+j];
        }
    }
    for (int i = 0; i < block_size; i++){
        for (int j = 0; j < dimy; j++){
            Q23_big[i*dimy+j] = Q2[i*dimy+j];
        }
    }
    for (int i = 0; i < block_size; i++){
        for (int j = 0; j < dimy; j++){
            Q23_big[(block_size+i)*dimy+j] = Q3[i*dimy+j];
        }
    }
    // Update Q01_big and Q23_big with the final merge factor Q02.
    void apply_merge_Q_big_update(double* Qsub, int rows, double* Qmerge, int ncol, int is_top) {
        int r_offset = is_top ? 0 : ncol;
        double* merge_sub = (double*) malloc(ncol * ncol * sizeof(double));
        if (!merge_sub) {
            printf("apply_merge_Q_big_update: allocation failed.\n");
            return;
        }
        for (int i = 0; i < ncol; i++){
            for (int j = 0; j < ncol; j++){
                merge_sub[i*ncol+j] = Qmerge[(r_offset+i)*ncol+j];
            }
        }
        double* result = (double*) calloc(rows * ncol, sizeof(double));
        if (!result) {
            printf("apply_merge_Q_big_update: allocation failed for result.\n");
            free(merge_sub);
            return;
        }
        /* Matrix matrix multiplication, no transpose. */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    rows, ncol, ncol,
                    1.0, Qsub, ncol,
                    merge_sub, ncol,
                    0.0, result, ncol);
        for (int i = 0; i < rows * ncol; i++){
            Qsub[i] = result[i];
        }
        free(merge_sub);
        free(result);
    }
    apply_merge_Q_big_update(Q01_big, 2*block_size, Q02, dimy, 1);
    apply_merge_Q_big_update(Q23_big, 2*block_size, Q02, dimy, 0);
    // Assemble final Q (economy size) from Q01_big and Q23_big.
    // Final Q will be (dimx x dimy) with dimx = 4*block_size.
    double *Q_final = (double*) malloc(dimx * dimy * sizeof(double));
    if (!Q_final) {
        printf("Allocation failed for Q_final.\n");
        return 1;
    }
    for (int i = 0; i < block_size; i++){
        for (int j = 0; j < dimy; j++){
            Q_final[i*dimy+j] = Q01_big[i*dimy+j];
        }
    }
    for (int i = 0; i < block_size; i++){
        for (int j = 0; j < dimy; j++){
            Q_final[(block_size+i)*dimy+j] = Q01_big[(block_size+i)*dimy+j];
        }
    }
    for (int i = 0; i < block_size; i++){
        for (int j = 0; j < dimy; j++){
            Q_final[(2*block_size+i)*dimy+j] = Q23_big[i*dimy+j];
        }
    }
    for (int i = 0; i < block_size; i++){
        for (int j = 0; j < dimy; j++){
            Q_final[(3*block_size+i)*dimy+j] = Q23_big[(block_size+i)*dimy+j];
        }
    }
    print_matrix("Final Q (dimx x dimy)", Q_final, dimx, dimy);
    print_matrix("Final R (dimy x dimy)", R02, dimy, dimy);
    
    // Stop timer for TSQR procedure and print elapsed time.
    double tsqr_end = omp_get_wtime();
    printf("\nTSQR execution time: %f seconds\n", tsqr_end - tsqr_start);
    
    // --- Compute Residuals and Orthogonality ---
    // 1. Compute A_approx = Q_final * R02 using BLAS.
    double *A_approx = (double*) malloc(dimx * dimy * sizeof(double));
    if (!A_approx) {
        printf("Allocation failed for A_approx.\n");
        return 1;
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                dimx, dimy, dimy,
                1.0, Q_final, dimy,
                R02, dimy,
                0.0, A_approx, dimy);
    double abs_error = frobenius_norm_diff(A, A_approx, dimx, dimy);
    double A_norm = frobenius_norm(A, dimx, dimy);
    double rel_error = abs_error / A_norm;
    printf("\nFrobenius norm of A: %e\n", A_norm);
    printf("Frobenius norm of (A - Q*R): %e\n", abs_error);
    printf("Relative error: %e\n", rel_error);
    free(A_approx);
    
    // 2. Check orthogonality of final Q.
    check_orthogonality(Q_final, dimx, dimy);
    
    // 3. For reference, compute a direct (full) QR factorization on A.
    double *A_copy = (double*) malloc(dimx * dimy * sizeof(double));
    if (!A_copy) {
        printf("Allocation failed for A_copy.\n");
        return 1;
    }
    memcpy(A_copy, A, dimx * dimy * sizeof(double));
    double *tau = (double*) malloc(dimy * sizeof(double));
    if (!tau) {
        printf("Allocation failed for tau.\n");
        free(A_copy);
        return 1;
    }
    int info_full = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, dimx, dimy, A_copy, dimy, tau);
    if (info_full != 0) {
        printf("Full QR: dgeqrf failed with info=%d.\n", info_full);
    }
    // Extract R_full from the upper triangle of A_copy.
    double *R_full = (double*) calloc(dimy * dimy, sizeof(double));
    for (int i = 0; i < dimx && i < dimy; i++){
        for (int j = i; j < dimy; j++){
            R_full[i*dimy+j] = A_copy[i*dimy+j];
        }
    }
    info_full = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, dimx, dimy, dimy, A_copy, dimy, tau);
    if (info_full != 0) {
        printf("Full QR: dorgqr failed with info=%d.\n", info_full);
    }
    // Now A_copy holds Q_full (dimx x dimy).
    double *A_approx_full = (double*) malloc(dimx * dimy * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                dimx, dimy, dimy,
                1.0, A_copy, dimy,
                R_full, dimy,
                0.0, A_approx_full, dimy);
    double full_abs_error = frobenius_norm_diff(A, A_approx_full, dimx, dimy);
    double full_rel_error = full_abs_error / frobenius_norm(A, dimx, dimy);
    printf("\nDirect full QR factorization residual:\n");
    printf("Frobenius norm of (A - Q_full*R_full): %e\n", full_abs_error);
    printf("Relative error: %e\n", full_rel_error);
    free(A_copy);
    free(tau);
    free(R_full);
    free(A_approx_full);
    // --- End New Section ---
    
    // Clean up and free mallocs 
    free(A);
    free(Q0); free(Q1); free(Q2); free(Q3);
    free(R0); free(R1); free(R2); free(R3);
    free(Q01); free(R01);
    free(Q23); free(R23);
    free(Q02); free(R02);
    free(Q01_big); free(Q23_big);
    free(Q_final);
    return 0;
}
