# Linen: A comfortable evolution of Flax

| 🚧  NOTE: Linen is in alpha. Parts of the API will change. Some people have started using Linen, but we're still [collecting feedback](https://github.com/google/flax/discussions) before finalizing the API and moving more users over. 🚧 |
| --- |

Linen is a rewrite of Flax Modules based on learning from our users and the broader JAX community. Linen addresses many complaints with our current `flax.nn` API. 
Moreover, Linen build on a new "functional core", enabling direct usage of JAX transformations such as `vmap`, `remat` or `scan` inside your modules.
  
In Linen, Modules behave much closer to vanilla Python objects, while still letting you opt-in to the concise single-method pattern many of our users love.

We're still working on documentation, but here are some references we have available now:
* 2-page intro to the [Linen Design Principles](https://docs.google.com/document/d/1ZlL_4bXCw5Xl0WstQw1GpnZqfb9JFOeUGAPcBVk-kn8)
* [Slides from a talk to the JAX core team](https://docs.google.com/presentation/d/1ngKWUwsSqAwPRvATG8sAxMzu9ujv4N__cKsUofdNno0)
* [Brief Intro to Linen](https://colab.sandbox.google.com/drive/1KPkQ_Hee9HlveynIF0FgPBzZDK79uoWL) in Colab
* An [upgrade guide](https://docs.google.com/document/d/1hYavTVPaKVVe9Be8pCB7yW7r6dDv3RALVNit8NZca4c) + some additional questions we're considering
* Ported [examples](https://github.com/google/flax/tree/master/linen_examples)
* "Design tests" used to ensure that our "functional core" supports [various advanced use-cases](https://github.com/google/flax/tree/master/linen_examples/core_design_test), and that the mostly-syntactic-sugar Module abstraction
  [doesn't get in the way](https://github.com/google/flax/tree/master/linen_examples/linen_design_test)