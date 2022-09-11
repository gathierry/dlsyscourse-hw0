#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (foat *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of exmaples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    // To get row i, col j: X[i * ncol + j]
    float batch_float = static_cast<float>(batch);
    size_t start = 0;
    while (start + batch <= m) {
        // exp_logits = np.exp(X_batch @ theta)
        // [b, n] * [n, k] -> [b, k]
        float exp_logits[batch * k] = {};
        for (size_t bi = 0; bi < batch; bi ++) {
            for (size_t ki = 0; ki < k; ki ++) {
                for (size_t ni = 0; ni < n; ni ++) {
                    exp_logits[bi * k + ki] += 
                        X[start * n + bi * n + ni] * theta[ni * k + ki];
                }
            }
        }
        for (size_t bi = 0; bi < batch; bi ++) {
            for (size_t ki = 0; ki < k; ki ++) {
                exp_logits[bi * k + ki] = exp(exp_logits[bi * k + ki]);
            }
        }
        // Z = exp_logits / np.sum(exp_logits, axis=1, keepdims=True) -> [b, k]
        float Z[batch * k] = {};
        for (size_t bi = 0; bi < batch; bi ++) {
            float exp_logits_sum = 0;
            for (size_t ki = 0; ki < k; ki ++) {
                exp_logits_sum += exp_logits[bi * k + ki];
            }
            for (size_t ki = 0; ki < k; ki ++) {
                Z[bi * k + ki] = exp_logits[bi * k + ki] / exp_logits_sum;
            }
        }
        // I_y = np.eye(theta.shape[1])[y_batch] -> [b, k]
        float I_y[batch * k] = {};
        for (size_t i = 0; i < batch; i ++) {
            I_y[i * k + y[start + i]] = 1;
        }
        // gradient = X_batch.T @ (Z - I_y) / batch -> [n, k]
        float gradient[n * k] = {};
        for (size_t ni = 0; ni < n; ni ++) {
            for (size_t ki = 0; ki < k; ki ++) {
                for (size_t bi = 0; bi < batch; bi ++) {
                    gradient[ni * k + ki] += 
                        X[start * n + bi * n + ni] 
                        * (Z[bi * k + ki] - I_y[bi * k + ki]) 
                        / batch_float;
                }
            }
        }
        // theta -= lr * gradient -> [n, k]
        for (size_t ni = 0; ni < n; ni ++) {
            for (size_t ki = 0; ki < k; ki ++) {
                theta[ni * k + ki] -= lr * gradient[ni * k + ki];
            }
        }
        start += batch;
    }
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
