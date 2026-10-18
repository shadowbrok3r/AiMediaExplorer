#[cfg(windows)]
extern crate winres;

fn main() {
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

/// Generates `icon_table.rs`: one `Icon` const per `assets/icons/regular/*.svg`.
fn generate_icon_table() -> Result<(), Box<dyn std::error::Error>> {
    use std::fmt::Write as _;
    use std::path::Path;
    use std::{env, fs};

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/icons/regular");
    println!("cargo:rerun-if-changed=assets/icons/duo");

    let manifest = env::var("CARGO_MANIFEST_DIR")?;
    let icons_dir = Path::new(&manifest).join("assets/icons");

    let mut names = Vec::new();
    for entry in fs::read_dir(icons_dir.join("regular"))? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "svg") {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("non-UTF-8 icon file name")?;
            names.push(stem.to_owned());
        }
    }
    names.sort();

    let mut out = String::new();
    let mut all = String::from("pub const ALL: &[Icon] = &[\n");
    let mut borrowed = String::from("pub(crate) const BORROWED: &[&str] = &[\n");
    for name in &names {
        let valid = name.starts_with(|c: char| c.is_ascii_lowercase())
            && name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
        if !valid {
            return Err(format!("icon file name `{name}.svg` must be lowercase kebab-case").into());
        }
        let ident = name.replace('-', "_").to_ascii_uppercase();
        let svg = fs::read_to_string(icons_dir.join("regular").join(format!("{name}.svg")))?;
        let glyph = match data_glyph(&svg) {
            Some(g) => {
                writeln!(borrowed, "    \"{name}\",")?;
                g.to_owned()
            }
            None if name.starts_with("ame-") => {
                return Err(format!(
                    "`{name}.svg` needs a data-glyph attribute naming a Phosphor glyph"
                )
                .into());
            }
            None => ident.clone(),
        };
        let duo = if icons_dir.join("duo").join(format!("{name}.svg")).is_file() {
            format!(
                "Some(include_bytes!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/assets/icons/duo/{name}.svg\")))"
            )
        } else {
            "None".to_owned()
        };
        writeln!(
            out,
            "pub const {ident}: Icon = Icon {{\n    name: \"{name}\",\n    uri: \"bytes://ame-icons/{name}.svg\",\n    \
             duo_uri: \"bytes://ame-icons/duo/{name}.svg\",\n    \
             svg: include_bytes!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/assets/icons/regular/{name}.svg\")),\n    \
             duo: {duo},\n    glyph: egui_phosphor::regular::{glyph},\n}};"
        )?;
        writeln!(all, "    {ident},")?;
    }
    all.push_str("];\n");
    borrowed.push_str("];\n");
    out.push_str(&all);
    out.push_str("/// Icons whose font glyph is another icon's.\n");
    out.push_str(&borrowed);

    fs::write(Path::new(&env::var("OUT_DIR")?).join("icon_table.rs"), out)?;
    Ok(())
}

/// Value of the root `data-glyph` attribute, if present.
fn data_glyph(svg: &str) -> Option<&str> {
    let start = svg.find("data-glyph=\"")? + "data-glyph=\"".len();
    let len = svg[start..].find('"')?;
    Some(&svg[start..start + len])
}