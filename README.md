# Quantum-Hybrid Convolutional NN with data re-uploading

Convolutional neural networks (hereinafter referred to as CNNs) have proven themselves as a good tool for solving the problem of image classification and recognition.

The process of training CNNs takes a lot of time, so recently the concept of building quantum-hybrid CNNs with the calculation of the convolution operation on a quantum computer has been actively gaining momentum.

At the moment, there are no quantum computers on which large images can be encoded, so I would like to create a quantum-hybrid CNN comparable in metrics to classical CNNs, where the image is initially processed by classical convolutional layers, and then by quantum.

## Existing solutions
- A fully quantum CNN has been created **[1]** - it uses variational parameters O(log(N)) for the input sizes of N qubits, which allows it to be effectively trained and implemented on realistic quantum devices of the near future, also post, quantum error correction has also been implemented.

- A quantum-hybrid CNN **[2]** has been created - it uses only 4 qubits, the concept of data re-uploading **[3]** has been implemented, which consists of re-encoding an image onto a quantum circuit, which often improves the final result.

Alas, none of the articles comes with an implementation on any framework.

## Purposes

- **Target**:
    - Create a quantum-hybrid CNN on a quantum emulator, comparable in metric (accuracy) to the classical CNN on small-sized images, using and expanding the approaches described in the article about quantum-hybrid CNN **[2]**
- **Tasks**:
    - Find a suitable dataset for experiments.
    - Implement a classic convolutional layer in pytorch.
    - Conduct experiments with different types of quantum layer models on pennylane.
    - Conduct experiments with resistance to quantum noise.

## Dataset and classical convolution
- For data we take the classic MNIST dataset of handwritten digits. We select from them only pictures with 0 and 1. For further experiments, we first change the pictures (initially 28x28x1 in size), reducing the size to 9x7x1 as in the original article, and also to 14x14x1.

- Describing the model class in pytorch, we define a convolutional layer with a kernel of size 3x3,
with a shift of 2 and zero padding, as well as with a 4x4 kernel and a shift of 2. As a result, we get pictures of sizes 4x3 and 6x6, respectively.

## Implementations of quantum layers
- For the first experiment, the ansatz proposed in the mentioned article was implemented.

![image](https://github.com/Nikait/qcnn_practice/assets/50284221/413cfa73-247f-4eee-af3c-eaceb02b188e)


The image (4x3) obtained at the output of the usual convolution is encoded line by line into 4 qubits, carrying out three rotations on each, respectively, after a series of two-qubit operators, the same data is loaded again, but with rotations in a different order. The last three parameters are trainable.

**Results:**

![image](https://github.com/Nikait/qcnn_practice/assets/50284221/9fde3601-0c14-40e9-a895-17edba53343f)


As a result, the algorithm converged, obtaining 96% quality on the test sample. Above is a graph of epochs versus quality.

- For the 2nd experiment we want to try loading a larger image so that we can initially work with larger images and increasingly transfer calculations to the quantum layer, thereby significantly reducing the running time of the entire algorithm.

We are introducing a new circuit with 12 qubits, onto which 6x6 pictures are loaded sequentially in a similar way, also with three trainable parameters at the end and measurement of the first qubit.

![image](https://github.com/Nikait/qcnn_practice/assets/50284221/75e931a5-265e-4842-8000-18cbd16b62b5)

With such ansatz we getting 0.912 accuracy:

![image](https://github.com/Nikait/QHCNN/assets/50284221/c96dab81-48bd-4366-9d0f-d6ab8117d3a3)

## Experiments with quantum noise

To realistically simulate the machines on which our circuits will run, as well as the robustness of the algorithms themselves, we need to add noise to our simulations.

There are many ways to do this, it was decided to use depolarization with probability p.

Results with some p values on the second version of the quantum layer are below:

|<div style="width:70px">p</div>|    accuracy   |
| ----------------------------- |:-------------:|
|              0                |     0.912     |
|            0.005              |     0.887     |
|            0.010              |     0.875     |
|            0.050              |     0.825     |

We can note that the algorithm turned out to be stable.
If p was too large, such as 0.05, convergence was no longer observed; for it, the table shows the accuracy at the peak at some point in the training.

More detailed graphics you can see in jupyter notebooks at /deprecated directory.


### installing dependencies
```console
foo@bar:~$ pip3 install -r requirements.txt
```
### run
```console
foo@bar:~$ python3 main.py
```

-----

**[1]** [Quantum Convolutional Neural Networks](https://arxiv.org/abs/1810.03787) \
        Iris Cong, Soonwon Choi, Mikhail D. Lukin, 2019
    
**[2]** [Hybrid quantum learning with data re-uploading on a small-scale superconducting quantum simulator](https://arxiv.org/abs/2305.02956) \
        Aleksei Tolstobrov, Gleb Fedorov, Shtefan Sanduleanu, Shamil Kadyrmetov, Andrei Vasenin, Aleksey Bolgar, Daria Kalacheva, Viktor Lubsanov, Aleksandr Dorogov, Julia Zotova, Peter Shlykov, Aleksei Dmitriev, Konstantin Tikhonov, Oleg V. Astafiev, 2024
    
**[3]** [Data re-uploading for a universal quantum classifier](https://arxiv.org/abs/1907.02085) \
        Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, José I. Latorre, 2020


