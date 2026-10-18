# backdrop-blur (vendored)

Subset of the MasterTech fork of [abdu-benayad/backdrop-blur](https://github.com/abdu-benayad/backdrop-blur)
(egui 0.37), limited to the eframe-on-glow grab-pass path:

- `backdrop-blur-core` - material and geometry vocabulary, error model, backend seam.
- `backdrop-blur-glow` - OpenGL 3.3 / GLES 3.0 grab-pass backend.
- `backdrop-blur-egui` - egui adapter (`GrabPassRenderer`), `grab-pass` feature only.

Differences from the fork:

- `backdrop-blur-wgpu`, the `own-loop` adapter path and the wgpu dependencies are removed.
- The `gl-snapshots` GPU test tier (EGL harness, `khronos-egl`, `libloading`) is removed.
- Manifests carry explicit pins instead of workspace inheritance; `egui` is built with `default-features = false`.

Library sources are otherwise unchanged. Upstream's README is kept in `UPSTREAM-README.md`.
Licensed MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`).
