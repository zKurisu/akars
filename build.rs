use std::env;
use std::path::{Path, PathBuf};

const DEFAULT_TPU_SDK_DIR: &str = "toolchains/tpu-sdk-sg200x";

fn main() {
    println!("cargo:rerun-if-env-changed=AKARS_TPU_SDK_DIR");

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch != "riscv64" {
        return;
    }

    let manifest_dir = env_path("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let tpu_sdk =
        env_path("AKARS_TPU_SDK_DIR").unwrap_or_else(|| manifest_dir.join(DEFAULT_TPU_SDK_DIR));
    let tpu_sdk_include = tpu_sdk.join("include");
    let tpu_sdk_lib = tpu_sdk.join("lib");

    require_dir("TPU SDK include", &tpu_sdk_include);
    require_dir("TPU SDK library", &tpu_sdk_lib);

    println!(
        "cargo:rerun-if-changed={}",
        tpu_sdk_include.join("cviruntime.h").display()
    );
    println!("cargo:rustc-link-search=native={}", tpu_sdk_lib.display());
    println!("cargo:rustc-link-lib=dylib=cviruntime");
    println!("cargo:rustc-link-lib=dylib=cvikernel");
    println!("cargo:rustc-link-lib=dylib=stdc++");
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
