#[cfg(windows)]
extern crate winres;

fn main() {
    emit_ffmpeg_cfg();
    generate_icon_table().expect("icon table generation failed");
    #[cfg(windows)]
    {
        static_vcruntime::metabuild();
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo rustc -- -Ctarget-feature=+crt-static");
        #[cfg(not(debug_assertions))]
        {
            let mut res = winres::WindowsResource::new();
            res.set_manifest(r#"
                <assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
                <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
                    <security>
                        <requestedPrivileges>
                            <requestedExecutionLevel level="requireAdministrator" uiAccess="false" />
                        </requestedPrivileges>
                    </security>
                </trustInfo>
                </assembly>
            "#);
            res.compile().unwrap();
        }
    }
}

/// Sets `ame_ffmpeg` when ffmpeg-next is used: always on Linux, elsewhere only with the `ffmpeg` feature; `AME_NO_NATIVE_VIDEO` turns it off.
fn emit_ffmpeg_cfg() {
    println!("cargo:rustc-check-cfg=cfg(ame_ffmpeg)");
    println!("cargo:rerun-if-env-changed=AME_NO_NATIVE_VIDEO");
    let linux = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "linux");
    let wanted = linux || std::env::var_os("CARGO_FEATURE_FFMPEG").is_some();
    if wanted && std::env::var_os("AME_NO_NATIVE_VIDEO").is_none() {
        println!("cargo:rustc-cfg=ame_ffmpeg");
    }
}

/// Generates `icon_table.rs`: one `Icon` const per `assets/icons/regular/*.svg`.
fn generate_icon_table() -> Result<(), Box<dyn std::error::Error>> {
    use egui_icon_pack_build::IconPackBuild;

    IconPackBuild {
        uri_prefix: "ame-icons".into(),
        custom_prefix: Some("ame-".into()),
        glyph_path: Some("egui_phosphor::regular".into()),
        ..IconPackBuild::in_out_dir("assets/icons", "icon_table.rs")
    }
    .generate()?;
    Ok(())
}
