#include "optimizer.h"
#include "threadpool.h"
#include <future>
#include <vector>
#include <thread>

SGD::SGD(scalar lr) : learning_rate(lr) {}

void SGD::update(Tensor &param, const Tensor &grad) {
  for (int i = 0; i < param.rows * param.cols; ++i) {
    param.data[i] -= learning_rate * grad.data[i];
  }
}

Adam::Adam(scalar lr, scalar b1, scalar b2, scalar eps)
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), step_count(0) {}

void Adam::begin_step() { step_count++; }

void Adam::update(Tensor &param, const Tensor &grad) {
  if (m.find(param.id) == m.end()) {
    m[param.id] = Tensor(param.rows, param.cols);
    v[param.id] = Tensor(param.rows, param.cols);
    for (int i = 0; i < param.rows * param.cols; ++i) {
      m[param.id].data[i] = 0;
      v[param.id].data[i] = 0;
    }
  }

  int t = std::max(step_count, 1);
  scalar beta1_correction = 1.0f - std::pow((scalar)beta1, (scalar)t);
  scalar beta2_correction = 1.0f - std::pow((scalar)beta2, (scalar)t);

  Tensor &m_t = m[param.id];
  Tensor &v_t = v[param.id];

  int num_elements = param.rows * param.cols;
  int num_threads = std::thread::hardware_concurrency();
  int chunk = (num_elements + num_threads - 1) / num_threads;

  std::vector<std::future<void>> futures;
  for (int t_idx = 0; t_idx < num_threads; ++t_idx) {
    int start = t_idx * chunk;
    int end = std::min(start + chunk, num_elements);
    if (start >= end) break;

    futures.push_back(ThreadPool::getInstance().enqueue([this, &param, &grad, &m_t, &v_t, start, end, beta1_correction, beta2_correction]() {
      for (int i = start; i < end; ++i) {
        scalar g = grad.data[i];
        // Only clip if strictly necessary, or use a faster clip
        if (g > 10.0f) g = 10.0f;
        else if (g < -10.0f) g = -10.0f;

        m_t.data[i] = beta1 * m_t.data[i] + (1.0f - beta1) * g;
        v_t.data[i] = beta2 * v_t.data[i] + (1.0f - beta2) * g * g;

        scalar m_hat = m_t.data[i] / beta1_correction;
        scalar v_hat = v_t.data[i] / beta2_correction;
        param.data[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
      }
    }));
  }
  for (auto &f : futures) f.wait();
}

RMSprop::RMSprop(scalar lr, scalar rho, scalar eps)
    : learning_rate(lr), rho(rho), epsilon(eps) {}

void RMSprop::update(Tensor &param, const Tensor &grad) {
  if (v.find(param.id) == v.end()) {
    v[param.id] = Tensor(param.rows, param.cols);
    for (int i = 0; i < param.rows * param.cols; ++i) {
      v[param.id].data[i] = 0;
    }
  }

  Tensor &v_t = v[param.id];

  for (int i = 0; i < param.rows * param.cols; ++i) {
    v_t.data[i] = rho * v_t.data[i] + (1.0f - rho) * grad.data[i] * grad.data[i];
    param.data[i] -=
        learning_rate * grad.data[i] / (std::sqrt(v_t.data[i]) + epsilon);
  }
}
