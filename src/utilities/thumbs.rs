use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use image::DynamicImage;
use std::path::Path;

pub fn generate_image_thumb_data(path: &Path) -> Result<String, String> {
    log::debug!("[thumb] generating image thumb: {}", path.display());
    let img = image::open(path)
    .map_err(|e| { 
        log::warn!("[thumb] open image failed {}: {}", path.display(), e); e.to_string() 
    })?;
    let thumb = img.thumbnail(256, 256);
    let mut buf = Vec::new();
    thumb
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .map_err(|e| { log::warn!("[thumb] encode png failed {}: {}", path.display(), e); e.to_string() })?;
    let b64 = BASE64.encode(&buf);
    log::debug!("[thumb] image thumb success: {} ({} bytes)", path.display(), buf.len());
    Ok(format!("data:image/png;base64,{}", b64))
}

#[cfg(windows)]
pub fn generate_video_thumb_data(path: &Path) -> Result<String, String> {
    use std::os::windows::ffi::OsStrExt;
    use windows::{
        Win32::{
            Foundation::SIZE,
            System::Com::{COINIT_APARTMENTTHREADED, CoInitializeEx, IBindCtx},
            UI::Shell::{IShellItem, IShellItemImageFactory, SHCreateItemFromParsingName, SIIGBF},
        },
        core::{Interface, PCWSTR},
    };

    unsafe {
        CoInitializeEx(None, COINIT_APARTMENTTHREADED)
            .ok()
            .map_err(|e| format!("CoInitializeEx: {e}"))?;
        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let shell_item: IShellItem =
            SHCreateItemFromParsingName(PCWSTR(wide.as_ptr()), None::<&IBindCtx>)
                .map_err(|e| format!("SHCreateItemFromParsingName: {e}"))?;
        let factory: IShellItemImageFactory = shell_item
            .cast()
            .map_err(|e| format!("cast IShellItemImageFactory: {e}"))?;
        let hbmp: windows::Win32::Graphics::Gdi::HBITMAP = factory
            .GetImage(SIZE { cx: 256, cy: 256 }, SIIGBF(0))
            .map_err(|e| format!("GetImage: {e}"))?;
        let data = hbitmap_to_png_data_url(hbmp)?;
        log::warn!("[thumb] video thumb success: {}", path.display());
        Ok(data)
    }
}

#[cfg(windows)]
unsafe fn hbitmap_to_png_data_url(
    hbmp: windows::Win32::Graphics::Gdi::HBITMAP,
) -> Result<String, String> {
    use windows::Win32::Graphics::Gdi::*;
    let mut bmp = BITMAP::default();
    if unsafe {
        GetObjectW(
            HGDIOBJ(hbmp.0),
            std::mem::size_of::<BITMAP>() as i32,
            Some(&mut bmp as *mut _ as *mut _),
        )
    } == 0
    {
        let _ = unsafe { DeleteObject(HGDIOBJ(hbmp.0)) };
        return Err("GetObjectW failed".into());
    }
    let width = bmp.bmWidth as i32;
    let height = bmp.bmHeight as i32;
    let mut bi = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: width,
            biHeight: -height,
            biPlanes: 1,
            biBitCount: 32,
            biCompression: 0,
            biSizeImage: 0,
            biXPelsPerMeter: 0,
            biYPelsPerMeter: 0,
            biClrUsed: 0,
            biClrImportant: 0,
        },
        bmiColors: Default::default(),
    };
    let stride = (width * 4) as usize;
    let mut buffer = vec![0u8; (stride * height as usize) as usize];
    let hdc: HDC = unsafe { CreateCompatibleDC(None) };
    if hdc.0.is_null() {
        let _ = unsafe { DeleteObject(HGDIOBJ(hbmp.0)) };
        return Err("CreateCompatibleDC failed".into());
    }
    let _old = unsafe { SelectObject(hdc, HGDIOBJ(hbmp.0)) };
    let got = unsafe {
        GetDIBits(
            hdc,
            hbmp,
            0,
            height as u32,
            Some(buffer.as_mut_ptr() as *mut _),
            &mut bi as *mut _,
            DIB_RGB_COLORS,
        )
    };
    let _ = unsafe { DeleteDC(hdc) };
    let _ = unsafe { DeleteObject(HGDIOBJ(hbmp.0)) };
    if got == 0 {
        return Err("GetDIBits failed".into());
    }
    for px in buffer.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
    let img = image::RgbaImage::from_raw(width as u32, height as u32, buffer)
        .ok_or("rgba from raw failed")?;
    let mut png = Vec::new();
    DynamicImage::ImageRgba8(img)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    let b64 = BASE64.encode(&png);
    Ok(format!("data:image/png;base64,{}", b64))
}
