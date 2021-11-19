# Lid Driven Cavity PINN

This code was written together with Mr. Jiang Qinghua and Dr. Shu Chang of the National University of Singapore, in partial fulfilment of my Final Year Project.

For the project, I compare two ML approaches to modelling a lid-driven cavity flow. The first is a data-driven multi-layer perceptron (MLP) model, which is a naive black-box function approximation of the true underlying process. The second is a physics-inferred neural net- work (PINN), which incorporates the Navier–Stokes equations into the loss function for training.

It is shown that the PINN allows for qualitatively and quantitatively better recovery of the fluid flow in low-data regimes. With more data, the improvements of the PINN compared to the MLP decrease.

The PINN also allows essentially unsupervised training when set up correctly; in the scope of the project, we recover the pressure field qualitatively without including any pressure training points. When one pressure point is included in training, the field is quantitatively accurately predicted.

Please read the attached PDF for full details on project methodology and observations.

The accuracy in execution or prediction of the given code is not guaranteed. Use of the included codes subject to the usual citation requirements.
