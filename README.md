# Neural-Style-Transfer

Neural Style Transfer (NST) is a Deep Learning technique allowing to create a **new image** $G$ by considering an initial **content image** $C$ and a **style image** $S$. The idea is to recreate the content image but with the style of another. 

The easiest way to perform this task is to start with a pre-trained network as VGG-19, as suggest by [Gatys et al. (2015)](https://arxiv.org/abs/1508.06576), because it has alreay learned to recognize a variety of low and high level features. Since, in shallower layer the network tends to detect lower-level features, while in the deeper layers higher-level features, a solution is to chose an **intermediate layer** to detect both of them at the same time. This should ensure to obtain the most visually pleasing results after training the NST network.  Thefore for the **content_layer** we adopt this strategy. By contrast, for **style layer** is good to select many layers due to the fact it has been shown that in this way the network should achieve better results if we merge style cost coming from different layers.

It is good to emphasize that during training the networks tries to minimize the cost function: $$J(G) = \alpha J_{content} (C,G) + \beta J_{style}(S,G)$$ where $J_{content}(C,G)$ is the **content cost function, while $J_{style}(S,G)$ is the **style cost function**. 

The notebook is written by using Python relying on TensorFlow and Keras.
