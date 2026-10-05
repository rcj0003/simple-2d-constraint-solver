#include "../include/gauss_seidel_sle_solver.h"

#include <cmath>
#include <assert.h>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace {

// Products are added in index order so each partial sum matches a scalar
// `sum += a[i] * b[i]`. The multiplies themselves run two at a time.
double dotInOrder(const double *a, const double *b, int n) {
    double sum = 0.0;
    int i = 0;
#if defined(__SSE2__) && !defined(__FMA__)
    for (; i + 2 <= n; i += 2) {
        const __m128d prod = _mm_mul_pd(_mm_loadu_pd(a + i), _mm_loadu_pd(b + i));
        sum += _mm_cvtsd_f64(prod);
        sum += _mm_cvtsd_f64(_mm_unpackhi_pd(prod, prod));
    }
#endif
    for (; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

}

atg_scs::GaussSeidelSleSolver::GaussSeidelSleSolver()
    : atg_scs::SleSolver(true)
{
    m_maxIterations = 128;
    m_minDelta = 1E-1;

    m_M.initialize(1, 1);
}

atg_scs::GaussSeidelSleSolver::~GaussSeidelSleSolver() {
    m_M.destroy();
    m_reg.destroy();
}

bool atg_scs::GaussSeidelSleSolver::solve(
        SparseMatrix<3> &J,
        Matrix &W,
        Matrix &right,
        Matrix *previous,
        Matrix *result)
{
    const int n = right.getHeight();
    
    result->resize(1, n);

    if (previous != nullptr && previous->getHeight() == n) {
        result->set(previous);
    }

    J.rightScale(W, &m_reg);
    m_reg.multiplyTranspose(J, &m_M);

    for (int i = 0; i < m_maxIterations; ++i) {
        const double maxDelta = solveIteration(
                m_M,
                right,
                result,
                result);

        if (maxDelta < m_minDelta) {
            return true;
        }
    }

    return false;
}

bool atg_scs::GaussSeidelSleSolver::solveWithLimits(
    SparseMatrix<3> &J,
    Matrix &W,
    Matrix &right,
    Matrix &limits,
    Matrix *result,
    Matrix *previous)
{
    const int n = right.getHeight();
    if (result->getHeight() != n) {
        result->initialize(1, n);
    }

    if (previous != nullptr && previous->getHeight() == n) {
        result->set(previous);
    }

    J.rightScale(W, &m_reg);
    m_reg.multiplyTranspose(J, &m_M);

    for (int i = 0; i < m_maxIterations; ++i) {
        const double maxDelta = solveIteration(
            m_M,
            right,
            limits,
            result,
            result);

        if (maxDelta < m_minDelta) {
            return true;
        }
    }

    return false;
}

double atg_scs::GaussSeidelSleSolver::solveIteration(
        Matrix &left,
        Matrix &right,
        Matrix *k_next,
        Matrix *k)
{
    double maxDifference = 0.0;
    const int n = k->getHeight();
    const double *kNextData = k_next->packed();
    const double *kData = k->packed();

    for (int i = 0; i < n; ++i) {
        const double *row = left.row(i);
        const double s0 = dotInOrder(row, kNextData, i);
        const double s1 = dotInOrder(row + i + 1, kData + i + 1, n - i - 1);

        const double k_next_i =
            (1 / row[i]) * (right.packed()[i] - s0 - s1);

        const double min_k = std::fmax(1E-3, kData[i]);
        const double delta = (std::abs(k_next_i) - min_k) / min_k;
        maxDifference = (delta > maxDifference)
            ? delta
            : maxDifference;

        k_next->set(0, i, k_next_i);
    }

    return maxDifference;
}

double atg_scs::GaussSeidelSleSolver::solveIteration(
        Matrix &left,
        Matrix &right,
        Matrix &limits,
        Matrix *k_next,
        Matrix *k)
{
    double maxDifference = 0.0;
    const int n = k->getHeight();
    const double *kNextData = k_next->packed();
    const double *kData = k->packed();

    for (int i = 0; i < n; ++i) {
        const double *row = left.row(i);
        const double s0 = dotInOrder(row, kNextData, i);
        const double s1 = dotInOrder(row + i + 1, kData + i + 1, n - i - 1);

        const double k_next_i =
            (1 / row[i]) * (right.packed()[i] - s0 - s1);

        const double limitMin = limits.get(0, i);
        const double limitMax = limits.get(1, i);
        const double x = std::fmax(limitMin, std::fmin(limitMax, k_next_i));

        const double min_k = std::fmax(1E-3, std::abs(k->get(0, i)));
        const double delta = std::abs(x - k->get(0, i)) / min_k;
        maxDifference = (delta > maxDifference)
            ? delta
            : maxDifference;

        k_next->set(0, i, x);
    }

    return maxDifference;
}
