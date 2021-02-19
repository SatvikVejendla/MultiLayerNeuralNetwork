# MultiLayerNeuralNetwork

# Javascript and Python library currently in progress.


### Javascript Implementation



```
const NeuralNetwork = require('./Objects/NeuralNetwork.js');
let nn = new NeuralNetwork(2, 4, 1);
for(let i = 0; i < 1000; i++){
    nn.train(i, Math.sin(i));
}
var output = nn.feedforward(Math.PI)
console.log(output); //something close to 0
```


### Python Implementation


```
import neuralnetwork
import math

nn = neuralnetwork.NeuralNetwork(2, 4, 1)
for i in range(1000):
    nn.train(i, math.sin(i)
output = nn.feedforward(Math.PI)
print(output) # something close to 0
```


# Checklist

- Initialize library✅
- Create simple 1 hidden layer module✅
- Multiple Layers

# Credits

JS - Satvik Vejendla
PY - Arnav Nayak
