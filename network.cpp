
#include "network.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#define QUADRATIC_COST (0)
#define L2_REGULARIZATION (1)
#define activation sigmoid
#define activation_prime sigmoid_prime

namespace base {

#pragma pack(push)
#pragma pack(2)

typedef struct {
  uint16 magic;  // must be 0x4242
  float64 learning_rate;
  float64 weight_reg_coeff;
  uint32 mini_batch_size;
  uint32 epoch_count;
  uint64 layer_count;
} NEURAL_NETWORK_HEADER;

#pragma pack(pop)

Network::Network(std::ifstream *infile) { Load(infile); }

Network::Network(const std::string &filename) {
  std::string error;
  Load(filename, &error);
  if (error.length()) {
    std::cout << error << std::endl;
  }
}

Network::Network(const std::string &filename, uint32 epoch_count,
                 uint32 mini_batch_size, const std::vector<uint32> &layer_sizes,
                 float64 rate, float64 reg, std::vector<SampleSet> *samples,
                 std::vector<SampleSet> *test_data) {
  epoch_count_ = epoch_count;
  mini_batch_size_ = mini_batch_size;
  learning_rate_ = rate;
  weight_reg_ = reg;

  assert(layer_sizes.size() > 1);

  for (uint32 i = 1; i < layer_sizes.size(); i++) {
    // The size of each layer matrix equals the W x H, where
    // W is the size of the prevous layer, and H is the size
    // of the current layer.
    weights_.push_back(Matrixf(layer_sizes[i - 1], layer_sizes[i]));
    biases_.push_back(Vectorf(layer_sizes[i]));
    neurons_.push_back(Vectorf(layer_sizes[i]));
    neuron_zs_.push_back(Vectorf(layer_sizes[i]));

    biases_[biases_.size() - 1].Randomize();
    weights_[weights_.size() - 1].Randomize();
    neurons_[neurons_.size() - 1].Zero();
    neuron_zs_[neuron_zs_.size() - 1].Zero();
  }

  // Train our network, saving the result to disk. If test_data is provided,
  // we'll save the best epoch. Otherwise, we'll save the last epoch.
  StochasticGradientDescent(filename, samples, test_data);
}

uint32 Network::GetInputLayerSize() const {
  assert(!weights_.empty());
  return weights_[0].GetWidth();
}

uint32 Network::GetOutputLayerSize() const {
  assert(!biases_.empty());
  return biases_[biases_.size() - 1].GetWidth();
}

bool SaveVector(std::ofstream *out_stream, const Vectorf &v) {
  if (!out_stream || 0 == v.GetWidth()) {
    return false;
  }
  for (uint32 i = 0; i < v.GetWidth(); i++) {
    if (!out_stream->write((char *)&v[i], sizeof(float64))) {
      return false;
    }
  }
  return true;
}

bool LoadVector(std::ifstream *in_stream, Vectorf *v) {
  if (!in_stream || !v || 0 == v->GetWidth()) {
    return false;
  }
  for (uint32 i = 0; i < v->GetWidth(); i++) {
    if (!in_stream->read((char *)&(*v)[i], sizeof(float64))) {
      return false;
    }
  }
  return true;
}

void Network::Save(const std::string &filename, std::string *error) const {
  std::ofstream out_stream(filename, std::ios::out | std::ios::binary);
  Save(&out_stream, error);
}

void Network::Save(std::ofstream *outfile, std::string *error) const {
  NEURAL_NETWORK_HEADER header = {
      0x4242,  // magic number
      learning_rate_, weight_reg_, mini_batch_size_, epoch_count_, 0};

  // Add one layer to account for the input layer.
  header.layer_count = weights_.size() + 1;

  if (!outfile->write((char *)&header, sizeof(header))) {
    if (error) {
      *error = "Failed to write neural network header to disk.";
    }
    return;
  }

  // Write out the sizes of each layer, beginning with the
  // input layer, and following with the rest of the layers.
  uint32 input_layer_size = weights_[0].GetWidth();
  if (!outfile->write((char *)&input_layer_size, sizeof(uint32))) {
    if (error) {
      *error = "Failed to write layer sizes to disk.";
    }
    return;
  }

  for (auto &layer : biases_) {
    uint32 layer_size = layer.GetWidth();
    if (!outfile->write((char *)&layer_size, sizeof(uint32))) {
      if (error) {
        *error = "Failed to write layer sizes to disk.";
      }
      return;
    }
  }

  for (auto &layer_weights : weights_) {
    uint32 layer_height = layer_weights.GetHeight();
    for (uint32 row = 0; row < layer_height; row++) {
      if (!SaveVector(outfile, layer_weights[row])) {
        if (error) {
          *error = "Failed to write layer weights to disk.";
        }
        return;
      }
    }
  }

  for (auto &layer_biases : biases_) {
    if (!SaveVector(outfile, layer_biases)) {
      if (error) {
        *error = "Failed to write layer biases to disk.";
      }
      return;
    }
  }
}

void Network::Load(const std::string &filename, std::string *error) {
  std::ifstream in_stream(filename, ::std::ios::in | ::std::ios::binary);
  Load(&in_stream, error);
}

void Network::Load(std::ifstream *infile, std::string *error) {
  NEURAL_NETWORK_HEADER header;
  if (!infile->read((char *)&header, sizeof(header))) {
    if (error) {
      *error = "Failed to read neural network header from disk.";
    }
    return;
  }

  if (header.magic != 0x4242) {
    if (error) {
      *error = "Invalid neural network file format detected.";
    }
    return;
  }

  learning_rate_ = header.learning_rate;
  weight_reg_ = header.weight_reg_coeff;
  mini_batch_size_ = header.mini_batch_size;
  epoch_count_ = header.epoch_count;

  std::vector<uint32> layer_sizes(header.layer_count);
  for (uint32 i = 0; i < header.layer_count; i++) {
    if (!infile->read((char *)&layer_sizes[i], sizeof(uint32))) {
      if (error) {
        *error = "Failed to read neural network layer sizes from disk.";
      }
      return;
    }
  }

  for (uint32 i = 1; i < layer_sizes.size(); i++) {
    weights_.push_back(Matrixf(layer_sizes[i - 1], layer_sizes[i]));
    biases_.push_back(Vectorf(layer_sizes[i]));
    neurons_.push_back(Vectorf(layer_sizes[i]));
    neuron_zs_.push_back(Vectorf(layer_sizes[i]));
    neurons_[neurons_.size() - 1].Zero();
    neuron_zs_[neuron_zs_.size() - 1].Zero();
  }

  for (auto &layer_weights : weights_) {
    uint32 layer_height = layer_weights.GetHeight();
    for (uint32 row = 0; row < layer_height; row++) {
      if (!LoadVector(infile, &layer_weights[row])) {
        if (error) {
          *error = "Failed to read layer weights from disk.";
        }
        return;
      }
    }
  }

  for (auto &layer_biases : biases_) {
    if (!LoadVector(infile, &layer_biases)) {
      if (error) {
        *error = "Failed to read layer biases from disk.";
      }
      return;
    }
  }
}

Vectorf Network::FeedForward(const Vectorf &sample) {
  // Prime the first iteration using our input layer.
  neuron_zs_[0] = weights_[0] * sample + biases_[0];
  neurons_[0] = activation(neuron_zs_[0]);
  // Propagate our neural activations to the output layer.
  for (uint32 i = 1; i < weights_.size(); i++) {
    neuron_zs_[i] = weights_[i] * neurons_[i - 1] + biases_[i];
    neurons_[i] = activation(neuron_zs_[i]);
  }

  return neurons_[neurons_.size() - 1];
}

float64 Network::Evaluate(std::vector<SampleSet> *test_data) {
  float64 total_correct = 0.0f;
  for (auto &data : *test_data) {
    Vectorf result = FeedForward(data.sample);
    uint32 network_guess = result.GetMaxIndex();
    uint32 ground_truth = data.label.GetMaxIndex();
    if (network_guess == ground_truth) {
      total_correct += 1.0;
    }
  }
  return total_correct;
}

void Network::StochasticGradientDescent(const std::string &filename,
                                        std::vector<SampleSet> *samples,
                                        std::vector<SampleSet> *test_data) {
  float64 total_samples_trained = 0.0;
  float64 highest_accuracy = -1.0;

  for (uint32 epoch = 0; epoch < epoch_count_; epoch++) {
    // For each epoch, shuffle our input samples, then operate in mini batch
    // sizes
    uint32 random_seed =
        std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(samples->begin(), samples->end(),
                 std::default_random_engine(random_seed));
    uint32 mini_batch_count = samples->size() / mini_batch_size_ +
                              !!(samples->size() % mini_batch_size_);

    // Allocate our gradient accumulators according to the shape of our
    // layer weights and biases.
    std::vector<Matrixf> grad_accum_weights = weights_;
    std::vector<Vectorf> grad_accum_biases = biases_;
    std::vector<Matrixf> grad_impulse_weights = weights_;
    std::vector<Vectorf> grad_impulse_biases = biases_;

    for (uint32 batch = 0; batch < mini_batch_count; batch++) {
      uint32 start_sample = batch * mini_batch_size_;
      uint32 end_sample = ((batch + 1) * mini_batch_size_ < samples->size())
                              ? (batch + 1) * mini_batch_size_
                              : samples->size();
      uint32 mini_batch_sample_count = end_sample - start_sample;

      for (auto &v : grad_accum_biases) v.Zero();
      for (auto &m : grad_accum_weights) m.Zero();

      // For each mini-batch, feed it through our network (feed forward), and
      // then compute the cost gradient (back propagation).
      for (uint32 s = start_sample; s < end_sample; s++) {
        FeedForward(samples->at(s).sample);

        for (auto &v : grad_impulse_biases) v.Zero();
        for (auto &m : grad_impulse_weights) m.Zero();

        BackPropagation(samples->at(s).sample, samples->at(s).label,
                        &grad_impulse_weights, &grad_impulse_biases);

        // Add our impulse gradient to our accumulation.
        for (uint32 layer = 0; layer < weights_.size(); layer++) {
          grad_accum_weights[layer] += grad_impulse_weights[layer];
          grad_accum_biases[layer] += grad_impulse_biases[layer];
        }
      }

      // Compute the average weights and biases of our negative gradient and
      // apply them to our actual network weights and biases.
      float64 divisor = (learning_rate_ / mini_batch_sample_count);
      // float64 reg_factor =
      //    (1.0 - learning_rate_ * weight_reg_ / mini_batch_sample_count);
      for (uint32 layer = 0; layer < weights_.size(); layer++) {
#if QUADRATIC_COST
#if L2_REGULARIZATION
        weights_[layer] =
            weights_[layer] * weight_reg_ - grad_accum_weights[layer] * divisor;
#else
        weights_[layer] = weights_[layer] - grad_accum_weights[layer] * divisor;
#endif
        biases_[layer] = biases_[layer] - grad_accum_biases[layer] * divisor;
#else
#if L2_REGULARIZATION
        weights_[layer] = weights_[layer] * weight_reg_ -
                          grad_accum_weights[layer] * divisor * -1.0;
#else
        weights_[layer] =
            weights_[layer] - grad_accum_weights[layer] * divisor * -1.0;
#endif
        biases_[layer] =
            biases_[layer] - grad_accum_biases[layer] * divisor * -1.0;
#endif
      }

      total_samples_trained += mini_batch_sample_count;
    }

    if (test_data) {
      float64 total_correct = Evaluate(test_data);
      std::cout << "Epoch " << epoch
                << " accuracy: " << 100.0f * total_correct / test_data->size()
                << "." << std::endl;
      if (total_correct > highest_accuracy) {
        highest_accuracy = total_correct;
        Save(filename);
      }
    } else {
      // If we have no test criteria then we simply write out the results
      // of every epoch.
      Save(filename);
    }
  }
  std::cout << std::endl;
}

void Network::BackPropagation(const Vectorf &sample, const Vectorf &label,
                              std::vector<Matrixf> *weight_gradient,
                              std::vector<Vectorf> *bias_gradient) {
  assert(bias_gradient->size() == biases_.size());
  assert(weight_gradient->size() == weights_.size());
  assert(weights_.size() == biases_.size());

  // Beginning with the final layer, compute gradient and store in
  // weight_gradient[last] and bias_gradient[last] Iterate backwards, using the
  // previous layer neurons, the weights and biases, and the derivatives from
  // the next layer.

  // Allocate a network to store activation derivatives.
  std::vector<Vectorf> dCdA = biases_;
  std::vector<Vectorf> dAdZ = biases_;

  for (auto &v : dCdA) {
    v.Zero();
  }
  for (auto &v : dAdZ) {
    v.Zero();
  }

  // Compute our activation derivatives, generating gradient weights and biases
  // along the way.
  for (int32 layer = dCdA.size() - 1; layer > 0; layer--) {
    for (uint32 node = 0; node < dCdA[layer].GetWidth(); node++) {
      if (layer == dCdA.size() - 1) {
        // We're evaluating the final (output) layer of our network.
#if QUADRATIC_COST
        dCdA[layer][node] = 2 * (neurons_[layer][node] - label[node]);
#else
        dCdA[layer][node] = label[node] / (neurons_[layer][node]) -
                            (1.0 - label[node]) / (1 - neurons_[layer][node]);
#endif
      } else {
        // We're evaluating a hidden layer of our network.
        // compute dCdA[layer][node] as the sum of weights * dCdZ of the next
        // layer.
        dCdA[layer][node] = 0.0;
        for (uint32 next_layer_node = 0;
             next_layer_node < dCdA[layer + 1].GetWidth(); next_layer_node++) {
          dCdA[layer][node] += weights_[layer + 1][next_layer_node][node] *
                               dAdZ[layer + 1][next_layer_node] *
                               dCdA[layer + 1][next_layer_node];
        }
      }

      // Compute the Z gradient and cache it
      dAdZ[layer][node] = activation_prime(neuron_zs_[layer][node]);

      // Compute the weight and bias gradients off of the dAdZ we just computed
      // For all weights for this node, compute weight gradient = a(l-1) * dAdZ
      // of this node * dCdA of this node
      if (layer == 0) {
        // We're at the first hidden layer of our network, so we should source
        // from the input.
        uint32 node_weights = sample.GetWidth();
        for (uint32 weight = 0; weight < node_weights; weight++) {
          weight_gradient->at(layer)[node][weight] =
              sample[weight] * dAdZ[layer][node] * dCdA[layer][node];
        }
      } else {
        uint32 node_weights = weight_gradient->at(layer).GetWidth();
        // We're not at the first hidden layer, so our source is simply the
        // previous layer.
        for (uint32 weight = 0; weight < node_weights; weight++) {
          weight_gradient->at(layer)[node][weight] =
              neurons_[layer - 1][weight] * dAdZ[layer][node] *
              dCdA[layer][node];
        }
      }
      // For the sole bias for this node, compute bias gradient = 1 * dAdZ of
      // this node * dCdA of this node
      bias_gradient->at(layer)[node] = dAdZ[layer][node] * dCdA[layer][node];
    }
  }
}

}  // namespace base
