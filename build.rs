#[cfg(windows)]
extern crate winres;

fn main() {
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