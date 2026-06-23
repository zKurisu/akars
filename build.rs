use std::env;
use std::path::{Path, PathBuf};

const DEFAULT_TOOLCHAIN_DIR: &str = "toolchains/xuantie-v3.4.0";
const DEFAULT_TPU_SDK_DIR: &str = "toolchains/tpu-sdk-sg200x";

fn main() {
    println!("cargo:rerun-if-env-changed=AKARS_TOOLCHAIN_DIR");
    println!("cargo:rerun-if-env-changed=AKARS_TPU_SDK_DIR");
    println!("cargo:rerun-if-env-changed=AKARS_OPENCV_DIR");
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=AR");
    println!("cargo:rerun-if-changed=src/cv_bridge.cpp");

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch != "riscv64" {
        return;
    }

    let manifest_dir = env_path("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let toolchain =
        env_path("AKARS_TOOLCHAIN_DIR").unwrap_or_else(|| manifest_dir.join(DEFAULT_TOOLCHAIN_DIR));
    let tpu_sdk =
        env_path("AKARS_TPU_SDK_DIR").unwrap_or_else(|| manifest_dir.join(DEFAULT_TPU_SDK_DIR));
    let opencv = env_path("AKARS_OPENCV_DIR").unwrap_or_else(|| tpu_sdk.join("opencv"));

    let tpu_sdk_include = tpu_sdk.join("include");
    let tpu_sdk_lib = tpu_sdk.join("lib");
    let opencv_include = opencv.join("include");
    let opencv_lib = opencv.join("lib");

    require_dir("TPU SDK include", &tpu_sdk_include);
    require_dir("TPU SDK library", &tpu_sdk_lib);
    require_dir("OpenCV include", &opencv_include);
    require_dir("OpenCV library", &opencv_lib);

    println!(
        "cargo:rerun-if-changed={}",
        tpu_sdk_include.join("cviruntime.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        opencv_include.join("opencv2/core.hpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        opencv_include.join("opencv2/imgcodecs.hpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        opencv_include.join("opencv2/imgproc.hpp").display()
    );

    let cxx =
        env_path("CXX").unwrap_or_else(|| toolchain.join("bin/riscv64-unknown-linux-musl-g++"));
    let ar = env_path("AR").unwrap_or_else(|| toolchain.join("bin/riscv64-unknown-linux-musl-ar"));

    cc::Build::new()
        .cpp(true)
        .compiler(cxx)
        .archiver(ar)
        .std("c++11")
        .file("src/cv_bridge.cpp")
        .include(&opencv_include)
        .flag("-fPIC")
        .flag("-O2")
        .flag("-Wall")
        .flag("-Wextra")
        .compile("akars_cv_bridge");

    println!("cargo:rustc-link-search=native={}", tpu_sdk_lib.display());
    println!("cargo:rustc-link-search=native={}", opencv_lib.display());
    println!("cargo:rustc-link-lib=dylib=cviruntime");
    println!("cargo:rustc-link-lib=dylib=cvikernel");
    println!("cargo:rustc-link-lib=dylib=opencv_core");
    println!("cargo:rustc-link-lib=dylib=opencv_imgcodecs");
    println!("cargo:rustc-link-lib=dylib=opencv_imgproc");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=z");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=atomic");
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn require_dir(label: &str, path: &Path) {
    assert!(
        path.is_dir(),
        "{label} directory does not exist: {}",
        path.display()
    );
}
