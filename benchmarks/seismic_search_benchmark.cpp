#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "amaiss/index.h"
#include "amaiss/index_factory.h"
#include "amaiss/io/index_io.h"
#include "amaiss/seismic_index.h"
#include "amaiss/types.h"

namespace {

struct CSRMatrix {
    int64_t nrow;
    int64_t ncol;
    int64_t nnz;
    std::vector<amaiss::idx_t> indptr;
    std::vector<amaiss::term_t> indices;
    std::vector<float> data;
};

CSRMatrix read_csr(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open CSR file: " + path);
    }

    CSRMatrix m;
    int64_t sizes[3];
    f.read(reinterpret_cast<char*>(sizes), sizeof(sizes));
    m.nrow = sizes[0];
    m.ncol = sizes[1];
    m.nnz = sizes[2];

    std::vector<int64_t> indptr64(m.nrow + 1);
    f.read(reinterpret_cast<char*>(indptr64.data()),
           static_cast<std::streamsize>((m.nrow + 1) * sizeof(int64_t)));
    m.indptr.resize(m.nrow + 1);
    for (int64_t i = 0; i <= m.nrow; ++i) {
        m.indptr[i] = static_cast<amaiss::idx_t>(indptr64[i]);
    }

    std::vector<int32_t> indices32(m.nnz);
    f.read(reinterpret_cast<char*>(indices32.data()),
           static_cast<std::streamsize>(m.nnz * sizeof(int32_t)));
    m.indices.resize(m.nnz);
    for (int64_t i = 0; i < m.nnz; ++i) {
        m.indices[i] = static_cast<amaiss::term_t>(indices32[i]);
    }

    m.data.resize(m.nnz);
    f.read(reinterpret_cast<char*>(m.data.data()),
           static_cast<std::streamsize>(m.nnz * sizeof(float)));

    return m;
}

std::string get_env_or_die(const char* name) {
    const char* val = std::getenv(name);
    if (val == nullptr || val[0] == '\0') {
        throw std::runtime_error(std::string("Environment variable not set: ") +
                                 name);
    }
    return val;
}

struct SeismicIndexFixture {
    static SeismicIndexFixture& instance() {
        static SeismicIndexFixture inst;
        return inst;
    }

    amaiss::Index* index{nullptr};
    CSRMatrix query;

private:
    SeismicIndexFixture() {
        std::string data_path = get_env_or_die("AMAISS_DATA_CSR");
        std::string query_path = get_env_or_die("AMAISS_QUERY_CSR");
        std::string dat_path = data_path + ".seismic.dat";

        std::cout << "Loading query CSR: " << query_path << "\n";
        query = read_csr(query_path);
        std::cout << "  rows=" << query.nrow << " cols=" << query.ncol
                  << " nnz=" << query.nnz << "\n";

        std::ifstream dat_check(dat_path, std::ios::binary);
        if (dat_check.good()) {
            dat_check.close();
            std::cout << "Found pre-built index: " << dat_path << "\n";
            index = amaiss::read_index(const_cast<char*>(dat_path.c_str()));
        } else {
            std::cout << "Loading data CSR: " << data_path << "\n";
            CSRMatrix data = read_csr(data_path);
            std::cout << "  rows=" << data.nrow << " cols=" << data.ncol
                      << " nnz=" << data.nnz << "\n";

            std::string desc = "seismic,lambda=6000|beta=400|alpha=0.4";
            index = amaiss::index_factory(static_cast<int>(data.ncol),
                                          desc.c_str());

            std::cout << "Adding vectors...\n";
            index->add(static_cast<amaiss::idx_t>(data.nrow),
                       data.indptr.data(), data.indices.data(),
                       data.data.data());

            std::cout << "Building index...\n";
            index->build();

            std::cout << "Saving index to: " << dat_path << "\n";
            amaiss::write_index(index, const_cast<char*>(dat_path.c_str()));
        }

        std::cout << "Index ready. num_vectors=" << index->num_vectors()
                  << "\n";
    }
};

}  // namespace

static void BM_Seismic_Search(benchmark::State& state) {
    auto& fix = SeismicIndexFixture::instance();
    const int k = static_cast<int>(state.range(0));
    const int n_queries = static_cast<int>(fix.query.nrow);

    amaiss::SeismicSearchParameters params(/*cut=*/3, /*heap_factor=*/1.0F);

    std::vector<float> distances(static_cast<size_t>(n_queries * k));
    std::vector<amaiss::idx_t> labels(static_cast<size_t>(n_queries * k));

    for (auto _ : state) {
        fix.index->search(static_cast<amaiss::idx_t>(n_queries),
                          fix.query.indptr.data(), fix.query.indices.data(),
                          fix.query.data.data(), k, distances.data(),
                          labels.data(), &params);
        benchmark::DoNotOptimize(distances.data());
        benchmark::DoNotOptimize(labels.data());
    }

    state.SetItemsProcessed(state.iterations() * n_queries);
    state.counters["queries"] = n_queries;
    state.counters["k"] = k;
    state.counters["QPS"] =
        benchmark::Counter(static_cast<double>(n_queries),
                           benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_Seismic_Search)
    ->Arg(10)
    ->Unit(benchmark::kMillisecond)
    ->Repetitions(10);
BENCHMARK(BM_Seismic_Search)
    ->Arg(100)
    ->Unit(benchmark::kMillisecond)
    ->Repetitions(10);
