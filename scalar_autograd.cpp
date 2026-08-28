#include <cassert>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>
#include <utility>

enum class Operation {
  leaf, // feature, weight, bias
  add, // 
  subtract, // 
  multiply, //
};

struct Node {
  double data = 0.0; // result of forward computation
  double grad = 0.0;  // relative change in loss wrt to the current node
  Operation operation = Operation::leaf;
  std::vector<std::shared_ptr<Node>> parents; // inputs that produced this node

  explicit Node(double value) : data(value) {}
};

class Value {
public:
  Value(double data): node_(std::make_shared<Node>(data)) {}

  [[nodiscard]] double data() const noexcept {
    return node_->data;
  }

  [[nodiscard]] double grad() const noexcept {
    return node_->grad;
  }

  friend Value operator+(
    const Value& left,
    const Value& right
  );

  friend Value operator-(
    const Value& left,
    const Value& right
  );

  friend Value operator*(
    const Value& left,
    const Value& right
  );

  void backward() {
    std::unordered_set<Node*> visited;
    std::vector<Node*> topology;

    build_topology(node_.get(), visited, topology);

    node_->grad = 1.0;

    for (
      auto current = topology.rbegin();
      current != topology.rend();
      ++current
    ) {
      Node* output = *current;

      switch (output->operation) {
        case Operation::leaf:
          break;

        case Operation::add:
          output->parents[0]->grad += output->grad;
          output->parents[1]->grad += output->grad;
          break;

        case Operation::subtract:
          output->parents[0]->grad += output->grad;
          output->parents[1]->grad -= output->grad;
          break;

        case Operation::multiply:
          output->parents[0]->grad +=
            output->parents[1]->data *
            output->grad;

          output->parents[1]->grad +=
            output->parents[0]->data *
            output->grad;

          break;
      }
    }

  }

private:
  std::shared_ptr<Node> node_;

  explicit Value(std::shared_ptr<Node> node): node_(std::move(node)) {}

  static void build_topology(
    Node* node,
    std::unordered_set<Node*>& visited,
    std::vector<Node*>& topology
  ) {
    if (!visited.insert(node).second) {
      return;
    }
    for (const auto& parent: node->parents) {
      build_topology(
        parent.get(), 
        visited,
        topology
      );
    }
    topology.push_back(node);
  }

};

Value operator+(
  const Value& left,
  const Value& right
)  {
  auto node = std::make_shared<Node>(
    left.data() + right.data()
  );
  node->operation = Operation::add;
  node->parents = {left.node_, right.node_};

  return Value(std::move(node));
}

Value operator-(
  const Value& left,
  const Value& right
)  {
  auto node = std::make_shared<Node>(
    left.data() - right.data()
  );
  node->operation = Operation::subtract;
  node->parents = {left.node_, right.node_};

  return Value(std::move(node));
}

Value operator*(
  const Value& left,
  const Value& right
)  {
  auto node = std::make_shared<Node>(
    left.data() * right.data()
  );
  node->operation = Operation::multiply;
  node->parents = {left.node_, right.node_};

  return Value(std::move(node));
}

//                 / --> branch A --\
// shared value - |                   --> loss
//                 \ --> branch B --/

int main() {
  const Value scalar_prediction(3.5);
  const Value scalar_target(2.5);
  const Value scalar_residual = scalar_prediction - scalar_target;

  Value scalar_loss = scalar_residual * scalar_residual;
  scalar_loss.backward();

  assert(scalar_loss.grad() == 1.0);
  assert(scalar_residual.grad() == 2.0);
  assert(scalar_prediction.grad() == 2.0);
  assert(scalar_target.grad() == -2.0);

  const Value x1(4.0);
  const Value x2(3.0);
  const Value x3(2.0);

  const Value w1(0.5);
  const Value w2(-1.0);
  const Value w3(2.0);

  const Value model_bias(0.5);
  const Value model_target(2.5);

  const Value model_prediction =
    w1 * x1 +
    w2 * x2 +
    w3 * x3 +
    model_bias;

  const Value model_residual = 
    model_prediction - model_target;

  Value model_loss = model_residual * model_residual;
  model_loss.backward();

  assert(model_prediction.data() == 3.5);
  assert(model_residual.data() == 1.0);
  assert(model_loss.data() == 1.0);

  assert(model_loss.grad() == 1.0);
  assert(model_residual.grad() == 2.0);
  assert(model_prediction.grad() == 2.0);
  assert(model_target.grad() == -2.0);
  assert(model_bias.grad() == 2.0);

  assert(w1.grad() == 8.0);
  assert(w2.grad() == 6.0);
  assert(w3.grad() == 4.0);


  std::cout << "Success!\n";
}










