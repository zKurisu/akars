//! SG2002 pin-mux configuration via the kernel's `/dev/pinmux` device.
//!
//! On this board the non-console UARTs (UART1/2/3) are enabled in the device
//! tree but have **no** pinctrl, so their TX/RX pads default to another
//! function and the port is electrically silent even though `/dev/ttySN`
//! opens fine. The kernel exposes `/dev/pinmux`, which writes the SoC FMUX
//! registers (base `0x0300_1000`) from user space. Its text interface takes
//! one `"0xOFFSET VALUE"` line per pad; offsets are relative to the FMUX base.
//!
//! UART1 is muxed onto the JTAG pads:
//!   - `JTAG_CPU_TMS` @ FMUX 0x64 -> UART1_TX  (FSEL=6)
//!   - `JTAG_CPU_TCK` @ FMUX 0x68 -> UART1_RX  (FSEL=6)
//! Configuring this consumes the JTAG pins (debug JTAG becomes unavailable).
//!
//! UART2 is muxed onto GPIOA pads:
//!   - GPIOA28 @ FMUX 0x70 -> UART2_TX  (FSEL=2)
//!   - GPIOA29 @ FMUX 0x74 -> UART2_RX  (FSEL=2)
//!
//! UART3 is muxed onto GPIOP pads:
//!   - GPIOP19 @ FMUX 0xD4 -> UART3_TX  (FSEL=5)
//!   - GPIOP20 @ FMUX 0xD8 -> UART3_RX  (FSEL=5)
//! **Warning**: GPIOP18-21 default to SDIO (WiFi); enabling UART3 breaks WiFi.

use std::io::{self, Write};

const PINMUX_DEVICE: &str = "/dev/pinmux";

// ── UART1 (JTAG pads) ──────────────────────────────────────────────
const FMUX_JTAG_CPU_TMS: u32 = 0x64;
const FMUX_JTAG_CPU_TCK: u32 = 0x68;
const FSEL_UART1: u32 = 6;

// ── UART2 (GPIOA28/A29) ────────────────────────────────────────────
const FMUX_GPIOA28: u32 = 0x70;
const FMUX_GPIOA29: u32 = 0x74;
const FSEL_UART2: u32 = 2;

// ── UART3 (GPIOP19/P20) ────────────────────────────────────────────
const FMUX_GPIOP19: u32 = 0xD4;
const FMUX_GPIOP20: u32 = 0xD8;
const FSEL_UART3: u32 = 5;

/// Write one FMUX register through `/dev/pinmux` using its text interface.
fn write_fmux(offset: u32, value: u32) -> io::Result<()> {
    let mut dev = std::fs::OpenOptions::new()
        .write(true)
        .open(PINMUX_DEVICE)?;
    let line = format!("0x{offset:X} {value}");
    dev.write_all(line.as_bytes())?;
    Ok(())
}

/// Route the JTAG pads to UART1 TX/RX so `/dev/ttyS1` can actually drive the
/// ESP32 base controller. Safe to call once at startup before opening the port.
pub fn configure_uart1() -> io::Result<()> {
    write_fmux(FMUX_JTAG_CPU_TMS, FSEL_UART1)?;
    write_fmux(FMUX_JTAG_CPU_TCK, FSEL_UART1)?;
    Ok(())
}

/// Route GPIOA28/A29 to UART2 TX/RX for `/dev/ttyS2` (ZP10S servo arm).
pub fn configure_uart2() -> io::Result<()> {
    write_fmux(FMUX_GPIOA28, FSEL_UART2)?;
    write_fmux(FMUX_GPIOA29, FSEL_UART2)?;
    Ok(())
}

/// Route GPIOP19/P20 to UART3 TX/RX for `/dev/ttyS3` (motor ESP32).
///
/// **Warning**: GPIOP18-21 default to SDIO (WiFi); enabling UART3 will
/// break WiFi connectivity on this board.
pub fn configure_uart3() -> io::Result<()> {
    write_fmux(FMUX_GPIOP19, FSEL_UART3)?;
    write_fmux(FMUX_GPIOP20, FSEL_UART3)?;
    Ok(())
}

/// Configure pin-mux for whichever UART backs `device`, based on the `ttySN`
/// alias.
pub fn configure_for_device(device: &str) {
    let result = if device.ends_with("ttyS1") {
        configure_uart1()
    } else if device.ends_with("ttyS2") {
        configure_uart2()
    } else if device.ends_with("ttyS3") {
        configure_uart3()
    } else {
        eprintln!("[pinmux] no pinmux profile for {device}; assuming pads already muxed");
        return;
    };

    match result {
        Ok(()) => eprintln!("[pinmux] {device} pads routed"),
        Err(err) => eprintln!("[pinmux] failed to configure {device} via {PINMUX_DEVICE}: {err}"),
    }
}
