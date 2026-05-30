# ClutterField

> Recursive predictive neural radar fields for clutter-aware perception on CPU and edge systems.

ClutterField is an experimental research project exploring recursive neural field architectures for radar perception in cluttered environments using CPU-first and edge-oriented design principles.

The project investigates how persistent latent spatial memory, multiresolution neural fields, and innovation-driven recursive updates can improve weak-target saliency and clutter-aware perception under low-compute constraints.

ClutterField combines:

* sparse multiresolution hash encodings,
* lightweight neural networks,
* predictive latent scene memory,
* recursive innovation accumulation,
* and hybrid DSP + learned inference

to explore a new class of persistent radar world models optimized for edge deployment.

---

# Motivation

Modern neural field systems such as Instant-NGP and NeRF-based architectures achieve impressive scene representations, but are primarily optimized for:

* dense rendering workloads,
* GPU acceleration,
* offline reconstruction,
* and photorealistic visual synthesis.

Radar perception operates under fundamentally different constraints:

* sparse and noisy observations,
* clutter-dominated environments,
* temporal coherence,
* low-SNR targets,
* streaming inference,
* and deployment on low-power CPU/edge hardware.

ClutterField explores a different direction:

> Can recursive predictive neural fields maintain persistent radar scene memory and enhance weak target saliency through innovation-driven temporal accumulation?

Rather than reconstructing static scenes frame-by-frame, ClutterField investigates persistent latent radar representations that continuously adapt over time.

---

# Core Ideas

ClutterField focuses on:

* CPU-first neural field architectures
* Sparse multiresolution hash-grid encodings
* Predictive latent scene modeling
* Recursive temporal memory
* Innovation residual processing
* Clutter-aware target saliency
* Sparse local latent updates
* Low-memory and low-latency inference
* Edge and low-SWaP deployment

---

# Current Research Direction

Current experiments explore:

```text
Radar Observation
        ↓
Predictive Latent Scene Memory
        ↓
Expected Radar Return
        ↓
Innovation Residual
        ↓
Recursive Temporal Accumulation
        ↓
Weak Target Emergence
```

The current hypothesis is that weak radar targets may become more detectable through persistent innovation accumulation against a learned clutter manifold, rather than through instantaneous thresholding alone.

---

# Initial Research Goals

* Implement CPU-native multiresolution hash encodings
* Explore recursive latent radar memory
* Investigate innovation-based target saliency
* Develop synthetic clutter and radar simulation environments
* Study sparse temporal update mechanisms
* Benchmark streaming neural field inference on x86 and ARM
* Investigate quantized and INT8 neural field execution
* Explore motion-aware recursive innovation accumulation

---

# Roadmap

## Phase 1 — Neural Field Foundations

* [x] 2D synthetic occupancy learning
* [x] CPU multiresolution hash encoding
* [x] Tiny MLP inference
* [x] Synthetic clutter generation

## Phase 2 — Predictive Recursive Fields

* [x] Predictive clutter modeling
* [x] Innovation residual computation
* [x] Temporal innovation accumulation
* [ ] Motion-aware innovation transport
* [ ] Recursive latent updates
* [ ] Sparse local memory updates

## Phase 3 — Synthetic Radar Dynamics

* [ ] Range-Doppler simulation
* [ ] Temporal clutter evolution
* [ ] Low-RCS target simulation
* [ ] Motion-consistent innovation accumulation
* [ ] Streaming radar tensors

## Phase 4 — Clutter-Aware Radar Fields

* [ ] Learned clutter manifolds
* [ ] Predictive radar scene modeling
* [ ] Recursive clutter suppression
* [ ] Persistent weak-target emergence
* [ ] Hybrid DSP + neural inference

## Phase 5 — Edge Optimization

* [ ] INT8 quantization
* [ ] SIMD acceleration
* [ ] ARM optimization
* [ ] Sparse cache-aware memory layouts
* [ ] Embedded deployment

---

# Long-Term Vision

ClutterField explores a new class of radar perception systems that are:

* recursive rather than frame-based,
* predictive rather than reactive,
* sparse rather than dense,
* CPU-native rather than GPU-dependent,
* and optimized for persistent cluttered environments.

Potential applications include:

* edge radar perception,
* low-SWaP sensing systems,
* drone and low-observable target detection,
* robotics,
* autonomous platforms,
* persistent spatial intelligence,
* and adaptive radar world modeling.

---

# Status

Early-stage experimental research prototype focused on recursive predictive radar memory and innovation-driven perception.
