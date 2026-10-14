#include "../include/wann/NsgaSort.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace wann {

// -------------------------------------------------------------------------
// getFronts – fast non-dominated sort on 2 objectives.
// Exactly mirrors Python nsga_sort.py::getFronts.
// -------------------------------------------------------------------------
std::vector<std::vector<int>>
getFronts(const std::vector<std::vector<double>>& objVals)
{
    const int n = static_cast<int>(objVals.size());
    // For now we use the first two objectives (WANN always passes 2).
    auto v1 = [&](int i){ return objVals[i][0]; };
    auto v2 = [&](int i){ return objVals[i][1]; };

    // S[p] = set of individuals dominated by p.
    std::vector<std::vector<int>> S(n);
    // n[p] = number of individuals that dominate p.
    std::vector<int> domCount(n, 0);
    std::vector<int> rank(n, 0);

    std::vector<std::vector<int>> fronts(1);

    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            if (p == q) continue;
            // p dominates q (maximisation)?
            bool pDomQ = (v1(p) >  v1(q) && v2(p) >  v2(q))
                      || (v1(p) >= v1(q) && v2(p) >  v2(q))
                      || (v1(p) >  v1(q) && v2(p) >= v2(q));
            // q dominates p?
            bool qDomP = (v1(q) >  v1(p) && v2(q) >  v2(p))
                      || (v1(q) >= v1(p) && v2(q) >  v2(p))
                      || (v1(q) >  v1(p) && v2(q) >= v2(p));

            if (pDomQ) {
                S[p].push_back(q);
            } else if (qDomP) {
                ++domCount[p];
            }
        }
        if (domCount[p] == 0) {
            rank[p] = 0;
            fronts[0].push_back(p);
        }
    }

    int i = 0;
    while (!fronts[i].empty()) {
        std::vector<int> Q;
        for (int p : fronts[i]) {
            for (int q : S[p]) {
                --domCount[q];
                if (domCount[q] == 0) {
                    rank[q] = i + 1;
                    Q.push_back(q);
                }
            }
        }
        ++i;
        fronts.push_back(Q);
    }
    fronts.pop_back();  // remove trailing empty front
    return fronts;
}

// -------------------------------------------------------------------------
// np.argsort(v) (kind='quicksort', the default) for float64, as implemented by
// numpy's scalar `aquicksort` (numpy/_core/src/npysort/quicksort.cpp): median-of-3
// introsort that falls back to insertion sort for partitions of <= 17 elements and
// to heapsort when the recursion gets too deep; NaN sorts last.
//
// The order of *equal* keys is what this reproduces. It matters here: the
// crowding distance of the extreme points of a front is +inf, so they tie, and
// their relative order decides who gets the best ranks. Recent numpy versions
// replace this routine with a SIMD sort on some CPUs (different tie order);
// the original WANN code was written against this scalar one.
// -------------------------------------------------------------------------
namespace {
inline bool npLess(double a, double b) { return a < b || (b != b && a == a); }

void aheapsort(const double* v, long* tosort, long n) {
    long* a = tosort - 1;  // heapsort indexes from 1
    long i, j, l, tmp;
    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && npLess(v[a[j]], v[a[j + 1]])) j += 1;
            if (npLess(v[tmp], v[a[j]])) { a[i] = a[j]; i = j; j += j; }
            else break;
        }
        a[i] = tmp;
    }
    for (; n > 1;) {
        tmp = a[n]; a[n] = a[1]; n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && npLess(v[a[j]], v[a[j + 1]])) j++;
            if (npLess(v[tmp], v[a[j]])) { a[i] = a[j]; i = j; j += i; }
            else break;
        }
        a[i] = tmp;
    }
}

std::vector<int> argsortAsc(const std::vector<double>& vec) {
    const long num = static_cast<long>(vec.size());
    std::vector<long> idx(num);
    std::iota(idx.begin(), idx.end(), 0L);
    if (num > 1) {
        const double* v = vec.data();
        constexpr long SMALL_QUICKSORT = 16;
        long* pl = idx.data();
        long* pr = idx.data() + num - 1;
        long* stack[128];  long** sptr = stack;
        int   depth[128];  int*   psdepth = depth;
        int cdepth = 0;
        for (unsigned long u = static_cast<unsigned long>(num); u >>= 1;) ++cdepth;
        cdepth *= 2;
        long *pm, *pi, *pj, *pk, vi;

        for (;;) {
            if (cdepth < 0) {
                aheapsort(v, pl, pr - pl + 1);
                goto stack_pop;
            }
            while ((pr - pl) > SMALL_QUICKSORT) {
                pm = pl + ((pr - pl) >> 1);
                if (npLess(v[*pm], v[*pl])) std::swap(*pm, *pl);
                if (npLess(v[*pr], v[*pm])) std::swap(*pr, *pm);
                if (npLess(v[*pm], v[*pl])) std::swap(*pm, *pl);
                {
                    const double vp = v[*pm];
                    pi = pl;
                    pj = pr - 1;
                    std::swap(*pm, *pj);
                    for (;;) {
                        do { ++pi; } while (npLess(v[*pi], vp));
                        do { --pj; } while (npLess(vp, v[*pj]));
                        if (pi >= pj) break;
                        std::swap(*pi, *pj);
                    }
                }
                pk = pr - 1;
                std::swap(*pi, *pk);
                // push the largest partition on the stack
                if (pi - pl < pr - pi) {
                    *sptr++ = pi + 1;
                    *sptr++ = pr;
                    pr = pi - 1;
                } else {
                    *sptr++ = pl;
                    *sptr++ = pi - 1;
                    pl = pi + 1;
                }
                *psdepth++ = --cdepth;
            }

            // insertion sort
            for (pi = pl + 1; pi <= pr; ++pi) {
                vi = *pi;
                const double vp = v[vi];
                pj = pi;
                pk = pi - 1;
                while (pj > pl && npLess(vp, v[*pk])) *pj-- = *pk--;
                *pj = vi;
            }
        stack_pop:
            if (sptr == stack) break;
            pr = *(--sptr);
            pl = *(--sptr);
            cdepth = *(--psdepth);
        }
    }
    return std::vector<int>(idx.begin(), idx.end());
}
} // namespace

// -------------------------------------------------------------------------
// getCrowdingDist – crowding distance within a single front.
// Port of Python nsga_sort.py::getCrowdingDist, including its normalisation
//     crowd *= abs((1/sortedObj[-1] - sortedObj[0]))
// which (by operator precedence) is NOT 1/(max-min); it is applied to every
// entry, infinite ones included (inf * 0 = NaN, as in numpy).
// -------------------------------------------------------------------------
std::vector<double> getCrowdingDist(const std::vector<double>& objVector) {
    const int m = static_cast<int>(objVector.size());
    if (m == 0) return {};

    const std::vector<int> key = argsortAsc(objVector);
    std::vector<double> sorted(m);
    for (int i = 0; i < m; ++i) sorted[i] = objVector[key[i]];

    const double Inf = std::numeric_limits<double>::infinity();

    // Distance to the neighbours on either side; the edges have an infinite one.
    std::vector<double> crowd(m);
    for (int i = 0; i < m; ++i) {
        double prev = (i > 0)   ? sorted[i-1] : Inf;
        double next = (i < m-1) ? sorted[i+1] : Inf;
        crowd[i] = std::abs(sorted[i] - prev) + std::abs(sorted[i] - next);
    }

    if ((sorted[m-1] - sorted[0]) > 0.0) {
        const double f = std::abs(1.0 / sorted[m-1] - sorted[0]);
        for (double& c : crowd) c *= f;
    }

    // Restore original order.
    std::vector<double> dist(m);
    for (int i = 0; i < m; ++i) dist[key[i]] = crowd[i];
    return dist;
}

// -------------------------------------------------------------------------
// nsga_sort – returns rank for each individual (0 = best).
// -------------------------------------------------------------------------
std::vector<int> nsga_sort(const std::vector<std::vector<double>>& objVals) {
    auto fronts = getFronts(objVals);

    // Within each front, rank by crowding distance (descending = better).
    for (auto& front : fronts) {
        const int fSize = static_cast<int>(front.size());
        std::vector<double> x1(fSize), x2(fSize);
        for (int i = 0; i < fSize; ++i) {
            x1[i] = objVals[front[i]][0];
            x2[i] = objVals[front[i]][1];
        }
        std::vector<double> crowd = getCrowdingDist(x1);
        {
            auto crowd2 = getCrowdingDist(x2);
            for (int i = 0; i < fSize; ++i) crowd[i] += crowd2[i];
        }
        // Sort front indices by decreasing crowding distance.
        std::vector<double> negCrowd(fSize);
        for (int i = 0; i < fSize; ++i) negCrowd[i] = -crowd[i];
        const std::vector<int> order = argsortAsc(negCrowd);  // np.argsort(-crowdDist)
        std::vector<int> sorted(fSize);
        for (int i = 0; i < fSize; ++i) sorted[i] = front[order[i]];
        front = sorted;
    }

    // Flatten fronts into a single ordering.
    std::vector<int> tmp;
    tmp.reserve(objVals.size());
    for (const auto& front : fronts)
        for (int ind : front) tmp.push_back(ind);

    // rank[individual] = its position in the flat ordering.
    std::vector<int> rank(tmp.size());
    for (int pos = 0; pos < static_cast<int>(tmp.size()); ++pos)
        rank[tmp[pos]] = pos;
    return rank;
}

} // namespace wann
