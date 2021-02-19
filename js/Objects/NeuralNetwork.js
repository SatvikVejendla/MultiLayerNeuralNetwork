const Matrix = require("./Matrix");

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y) {
  return y * (1 - y);
}
class NeuralNetwork {
  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    this.weights = {
      weights_ih: new Matrix(this.hidden_nodes[0], this.input_nodes),
    };
    for (let i = 0; i < hidden_nodes.length; i++) {
      if (i == hidden_nodes.length - 1) {
        this.weights["weights_ho"] = new Matrix(
          this.output_nodes,
          this.hidden_nodes[i]
        );
      } else if (i >= 0) {
        this.weights["weights_h" + i] = new Matrix(
          this.hidden_nodes[i + 1],
          this.hidden_nodes[i]
        );
      }
    }
    for (let i in this.weights) {
      this.weights[i].randomize();
    }

    this.biases = {
      bias_ih: new Matrix(this.hidden_nodes[0], 1),
    };

    for (let i = 0; i < hidden_nodes.length; i++) {
      if (i == hidden_nodes.length - 1) {
        this.biases["bias_ho"] = new Matrix(this.output_nodes, 1);
      } else {
        this.biases["bias_h" + i] = new Matrix(this.hidden_nodes[i + 1], 1);
      }
    }
    for (let i in this.biases) {
      this.biases[i].randomize();
    }

    /*this.bias_h = new Matrix(this.hidden_nodes, 1);
    this.bias_o = new Matrix(this.output_nodes, 1);
    this.bias_h.randomize();
    this.bias_o.randomize();*/
    this.learning_rate = 0.1;
  }
  getWeights() {
    return this.weights;
  }

  /*getWeights(x) {
    if (x == 0) {
      return this.weights_ih;
    } else {
      return this.weights_ho;
    }
  }*/
  getBias() {
    return this.biases;
  }

  feedforward(input_array) {
    let inputs = Matrix.fromArray(input_array);

    //input to hidden
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);

    //activation
    hidden.map(sigmoid);

    for (let i = 1; i < this.weights.length - 1; i++) {
      let tempoutput = Matrix.multiply(this.weights["weights_h" + i], hidden);
      hidden.add(this.biases["bias_h" + i]);
      hidden.map(sigmoid);
    }

    //hidden to output
    let output = Matrix.multiply(this.weights_ho, hidden);
    output.add(this.bias_o);
    output.map(sigmoid);

    //converting matrix to array and return
    return output.toArray();
  }

  train(input_array, target_array) {
    let inputs = Matrix.fromArray(input_array);

    //input to hidden
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);

    //activation
    hidden.map(sigmoid);

    //hidden to output
    let outputs = Matrix.multiply(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(sigmoid);

    let targets = Matrix.fromArray(target_array);
    //error calculation
    let output_errors = Matrix.subtract(targets, outputs);
    let gradients = Matrix.map(outputs, dsigmoid);
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);

    let hidden_T = Matrix.transpose(hidden);
    let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);

    //Weights and Biases adjustments
    this.weights_ho.add(weight_ho_deltas);
    this.bias_o.add(gradients);

    //hidden layer errors
    let who_t = Matrix.transpose(this.weights_ho);
    let hidden_errors = Matrix.multiply(who_t, output_errors);
    let hidden_gradient = Matrix.map(hidden, dsigmoid);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);

    let inputs_T = Matrix.transpose(inputs);
    let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

    this.weights_ih.add(weight_ih_deltas);
    this.bias_h.add(hidden_gradient);
  }
}

let nn = new NeuralNetwork(2, [4, 2], 1);
var arr = nn.getBias();

//arr = Object.keys(arr).map((i) => arr[i].data);
console.log(arr);
