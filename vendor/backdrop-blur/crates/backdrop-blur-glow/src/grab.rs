//! The grab pass: copy a region of the **live** draw framebuffer into a sampleable RGBA8 texture
//! the blur can read (DESIGN §5/§11). This is the half of the seam the own-loop/wgpu backend never
//! needs — wgpu receives a ready intermediate; glow must extract one from whatever the host just
//! rendered into.
//!
//! Coordinates are GL bottom-left throughout (the region is a [`GlRegion`]): `copyTexSubImage2D` /
//! `blitFramebuffer` read from `region.origin_bl()` directly, because GL's read origin is the
//! framebuffer's bottom-left — **no flip** (DESIGN §5). The grabbed texture's row 0 is therefore
//! the framebuffer's bottom row, matching the bottom-left fullscreen-triangle vertex shader.

use crate::GlowBlur;
use backdrop_blur_core::{BlurError, BlurStage, GlRegion};
use glow::HasContext;

/// The cached grab target: a single-sample RGBA8 texture holding the grabbed region (the blur
/// samples it) plus an FBO with that texture attached (the MSAA blit destination). **Size-keyed** —
/// reallocated when the clamped region size changes, because `copyTexSubImage2D`/`blitFramebuffer`
/// do not allocate (a grow past a pre-sized texture is `GL_INVALID_VALUE`; IMPL §2c).
#[derive(Clone, Copy)]
pub(crate) struct GrabTarget {
    pub(crate) size: [u32; 2],
    /// The grab source the blur samples (RGBA8, holds the gamma-encoded framebuffer content).
    pub(crate) source: glow::Texture,
    /// `source` attached as color 0 — the destination for the MSAA-resolve blit.
    fbo: glow::Framebuffer,
}

impl GlowBlur {
    /// Grab `region` out of the live draw framebuffer `read_fb` into a sampleable RGBA8 texture.
    /// `read_fb` is whatever the caller captured from `GL_DRAW_FRAMEBUFFER_BINDING` (`None` = the
    /// default framebuffer 0). On an MSAA context (`profile.samples > 0`) the region is
    /// blit-resolved straight into the grab FBO (a same-size multisample resolve); otherwise it is
    /// copied with `copyTexSubImage2D`. Every GL binding the grab perturbs (read/draw FBO, the bound
    /// `TEXTURE_2D`) is saved and restored, so the host's state is left as found.
    pub(crate) fn grab(
        &mut self,
        gl: &glow::Context,
        read_fb: Option<glow::Framebuffer>,
        region: GlRegion,
    ) -> Result<glow::Texture, BlurError> {
        let target = self.ensure_grab_target(gl, region.size())?;
        // `profile.samples` is the DEFAULT-framebuffer sample count probed once at `new()`. The grab
        // assumes `read_fb` is that same default framebuffer (true for eframe/cage, which render the
        // host UI into the default FB and run the paint callback against it), so a `samples > 0`
        // context means the region must be MSAA-resolved before it is sampleable.
        let samples = self.profile.samples;
        let [ox, oy] = region.origin_bl();
        let [w, h] = region.size();
        debug_assert!(
            w > 0 && h > 0,
            "grab region must be non-empty (caller clipped via clip_to)"
        );
        let (ox, oy, w, h) = (ox as i32, oy as i32, w as i32, h as i32);

        // SAFETY: read-only queries of the bindings the grab will perturb, so they can be restored.
        let (saved_read, saved_draw, saved_tex) = unsafe {
            (
                gl.get_parameter_framebuffer(glow::READ_FRAMEBUFFER_BINDING),
                gl.get_parameter_framebuffer(glow::DRAW_FRAMEBUFFER_BINDING),
                gl.get_parameter_texture(glow::TEXTURE_BINDING_2D),
            )
        };

        // SAFETY: bind the captured live target as the read source and grab the region into our
        // target. `read_fb` is the caller's GL_DRAW_FRAMEBUFFER_BINDING; `target.{source,fbo}` were
        // created by `ensure_grab_target` on this current context. In-bounds reads are a CALLER
        // PRECONDITION: the egui adapter clips the region to the framebuffer (GlRegion::intersect/
        // clip_to, IMPL §3b) before grabbing; `grab_source` passes it through unclipped. An
        // out-of-range copyTexSubImage2D/blitFramebuffer read is GL-DEFINED — undefined *texels*
        // (zeros / clamped), not memory-unsafe — so a precondition miss is a wrong-pixels bug, not
        // UB. The bindings are restored before returning.
        unsafe {
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, read_fb);
            if samples > 0 {
                // MSAA: resolve the region directly into the grab texture (same-size resolve blit).
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(target.fbo));
                gl.blit_framebuffer(
                    ox,
                    oy,
                    ox + w,
                    oy + h,
                    0,
                    0,
                    w,
                    h,
                    glow::COLOR_BUFFER_BIT,
                    glow::NEAREST,
                );
            } else {
                // Single-sample: copy the region into the grab texture (origin_bl maps straight to
                // the GL read origin — no flip).
                gl.bind_texture(glow::TEXTURE_2D, Some(target.source));
                gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, ox, oy, w, h);
            }
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, saved_read);
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, saved_draw);
            gl.bind_texture(glow::TEXTURE_2D, saved_tex);
        }
        Ok(target.source)
    }

    /// The grab target sized to `size`, (re)created on a size change.
    fn ensure_grab_target(
        &mut self,
        gl: &glow::Context,
        size: [u32; 2],
    ) -> Result<GrabTarget, BlurError> {
        if let Some(t) = self.grab {
            if t.size == size {
                return Ok(t);
            }
            self.destroy_grab(gl);
        }
        let target = create_grab_target(gl, size)?;
        self.grab = Some(target);
        Ok(target)
    }

    /// Delete the cached grab target (on resize and from [`Self::destroy`]).
    pub(crate) fn destroy_grab(&mut self, gl: &glow::Context) {
        if let Some(t) = self.grab.take() {
            // SAFETY: both handles were created by `create_grab_target` on this current context and
            // are deleted exactly once (the `take` prevents a second delete).
            unsafe {
                gl.delete_framebuffer(t.fbo);
                gl.delete_texture(t.source);
            }
        }
    }
}

/// Create a `size`-sized RGBA8 grab texture (ClampToEdge, Linear) and an FBO with it attached.
fn create_grab_target(gl: &glow::Context, size: [u32; 2]) -> Result<GrabTarget, BlurError> {
    let w = size[0].max(1) as i32;
    let h = size[1].max(1) as i32;

    // SAFETY: `gl` is current; `create_texture` returns a fresh texture handle or an error string.
    let source = unsafe { gl.create_texture() }.map_err(|e| BlurError::ResourceCreation {
        stage: BlurStage::PingPongTexture,
        source: e.into(),
    })?;
    // SAFETY: `source` was just created; allocate RGBA8 storage and set ClampToEdge/Linear on it.
    // PixelUnpackData::Slice(None) allocates without uploading.
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(source));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            w,
            h,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(None),
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as i32,
        );
    }

    // SAFETY: `gl` is current; `create_framebuffer` returns a fresh FBO or an error string.
    let fbo = match unsafe { gl.create_framebuffer() } {
        Ok(fbo) => fbo,
        Err(e) => {
            // SAFETY: `source` was just created on this context and is otherwise unreferenced.
            unsafe { gl.delete_texture(source) };
            return Err(BlurError::ResourceCreation {
                stage: BlurStage::Framebuffer,
                source: e.into(),
            });
        }
    };
    // SAFETY: attach `source` to `fbo`, check completeness, then unbind. All handles are live on
    // this current context.
    let status = unsafe {
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(source),
            0,
        );
        let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        status
    };
    if status != glow::FRAMEBUFFER_COMPLETE {
        // SAFETY: both handles created above on this context; deleted once on the error path.
        unsafe {
            gl.delete_framebuffer(fbo);
            gl.delete_texture(source);
        }
        return Err(BlurError::ResourceCreation {
            stage: BlurStage::Framebuffer,
            source: format!("grab framebuffer incomplete: 0x{status:X}").into(),
        });
    }
    Ok(GrabTarget { size, source, fbo })
}
