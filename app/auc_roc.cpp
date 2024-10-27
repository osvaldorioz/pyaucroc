#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <vector>
#include <stdexcept>

//c++ -O3 -Ofast -Wall -shared -std=c++20 -fPIC `python3.12 -m pybind11 --includes` auc_roc.cpp -o auc_roc_module`python3.12-config --extension-suffix`

namespace py = pybind11;

double calculate_auc(const std::vector<double>& labels, const std::vector<double>& scores) {
    if (labels.size() != scores.size()) {
        throw std::invalid_argument("Labels and scores must have the same length");
    }

    // Crear un vector de pares (score, label) y ordenarlo en función de los scores en orden descendente
    std::vector<std::pair<double, double>> score_label_pairs;
    for (size_t i = 0; i < labels.size(); ++i) {
        score_label_pairs.emplace_back(scores[i], labels[i]);
    }
    std::sort(score_label_pairs.begin(), score_label_pairs.end(), std::greater<>());

    // Calcular los valores de AUC utilizando el método de trapezoides
    double auc = 0.0;
    double tp = 0.0;  // Verdaderos positivos
    double fp = 0.0;  // Falsos positivos
    double prev_tp = 0.0;
    double prev_fp = 0.0;

    for (const auto& pair : score_label_pairs) {
        if (pair.second == 1.0) {
            tp += 1.0;
        } else {
            fp += 1.0;
            // Añadir el área del trapecio
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
            prev_tp = tp;
            prev_fp = fp;
        }
    }

    // Dividir por el producto de verdaderos positivos y falsos positivos
    if (tp * fp == 0) {
        throw std::runtime_error("AUC is undefined due to no positive or negative samples");
    }
    auc /= (tp * fp);
    return auc;
}

py::object calculate_auc_py(py::array_t<double> labels, py::array_t<double> scores) {
    // Convertir los arrays numpy a vectores de C++
    auto labels_buf = labels.request();
    auto scores_buf = scores.request();
    if (labels_buf.size != scores_buf.size) {
        throw std::runtime_error("Labels and scores must have the same size");
    }

    std::vector<double> labels_vec(static_cast<double*>(labels_buf.ptr), static_cast<double*>(labels_buf.ptr) + labels_buf.size);
    std::vector<double> scores_vec(static_cast<double*>(scores_buf.ptr), static_cast<double*>(scores_buf.ptr) + scores_buf.size);

    // Calcular el AUC
    double auc = calculate_auc(labels_vec, scores_vec);
    return py::float_(auc);
}

PYBIND11_MODULE(auc_roc_module, m) {
    m.def("calculate_auc", &calculate_auc_py, "Calculate AUC-ROC from labels and scores");
}
