#ifndef DENSE_VECTOR_MATRIX_H
#define DENSE_VECTOR_MATRIX_H

#include <cstddef>
#include <cstdlib>

namespace amaiss {

class DenseVectorMatrix {
#ifdef __AVX512F__
    static constexpr int ALIGNMENT = 64;  // AVX512
#else
    static constexpr int ALIGNMENT = 16;
#endif

public:
    DenseVectorMatrix(const DenseVectorMatrix&) = delete;
    DenseVectorMatrix& operator=(const DenseVectorMatrix&) = delete;
    DenseVectorMatrix(DenseVectorMatrix&&) = delete;

    DenseVectorMatrix(size_t row, size_t dimension)
        : rows_(row), dimension_(dimension) {
        data_ = static_cast<float*>(
            std::aligned_alloc(ALIGNMENT, row * dimension * sizeof(float)));
    }

    ~DenseVectorMatrix() { std::free(data_); }

    float get(size_t row, size_t col) const {
        return data_[row * dimension_ + col];
    }

    void set(size_t row, size_t col, float value) {
        data_[row * dimension_ + col] = value;
    }

    float* data() const { return data_; }
    const size_t get_rows() const { return rows_; }
    const size_t get_dimension() const { return dimension_; }

private:
    float* data_;
    size_t rows_;
    size_t dimension_;
};

}  // namespace amaiss

#endif  // DENSE_VECTOR_MATRIX_H