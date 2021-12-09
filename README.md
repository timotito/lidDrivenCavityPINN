# Lid Driven Cavity PINN

This code was written together with Mr. Jiang Qinghua and Dr. Shu Chang, in partial fulfilment of my Final Year Project at the National University of Singapore (NUS). It is an extension of prior work by Maziar Riassi, and attempts to further Riassi's research by more critically examining the limited unsupervised learning aspect of Riassi's model. In practical terms, it is also an upgrade over Riassi's work, by enabling improvements brought on by using Tensorflow v2 over Riassi's use of TFv1.

For the project, I compare two ML approaches to modelling a lid-driven cavity flow. The first is a data-driven multi-layer perceptron (MLP) model, which is a naive black-box function approximation of the true underlying process. The second is a physics-inferred neural net- work (PINN), which incorporates the Navier–Stokes equations into the loss function for training.

It is shown that the PINN allows for qualitatively and quantitatively better recovery of the fluid flow in low-data regimes. With more data, the improvements of the PINN compared to the MLP decrease.

The PINN also allows essentially unsupervised training when set up correctly; in the scope of the project, we recover the pressure field qualitatively without including any pressure training points. When one pressure point is included in training, the field is quantitatively accurately predicted.

Please read the attached PDF for full details on project methodology and observations.

In `train_nn.py`, a brief minimum working example of the underlying code is provided. In the `PhysicsInformedNN` class, I define a custom loss function incorporating the standard MSE as well as additional MSE derived by testing the PINN against the Navier--Stokes equations. This file is meant to be run as-is, with minimal customisation required, in order to provide a basic idea of how the PINN works. Further modifications can be made to the code, to achieve more interesting results.

The training of the PINN is directly contrasted to the training of a standard naive neural network, internally named `mlp` for multi-layer perceptron. Both models are trained using the same observed data and the weights of the converged models are saved, allowing for subsequent analysis of MSEs and other statistical data.

The accuracy in execution or prediction of the given code is not guaranteed. Use of the included codes subject to the usual academic citation requirements.
