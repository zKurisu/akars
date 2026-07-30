//! Standalone camera web-view + basic drive/arm controls.
//!
//! Streams live camera feed to a web page and provides forward / backward /
//! left / right / grab / release buttons.
//!
//! Usage:
//!   ./cam_web
//!   ./cam_web --camera /dev/cvi-usb-camera0 --motor /dev/ttyS1 --arm /dev/ttyS2 --port 8080

use akars::arm::Arm;
use akars::camera::UsbCamera;
use akars::motor::{Motor, MotorConfig};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// ── Shared state ──────────────────────────────────────────────────────

struct AppState {
    camera_jpeg: Arc<Mutex<Option<Vec<u8>>>>,
    motor: Arc<Mutex<Motor>>,
    arm: Arc<Mutex<Arm>>,
}

// ── CLI / config ──────────────────────────────────────────────────────

struct Config {
    camera: String,
    motor: String,
    arm: String,
    port: u16,
}

fn main() {
    let mut cfg = Config {
        camera: "/dev/cvi-usb-camera0".into(),
        motor: "/dev/ttyS1".into(),
        arm: "/dev/ttyS2".into(),
        port: 8080,
    };
    {
        let mut args = std::env::args().skip(1).peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--camera" => cfg.camera = args.next().unwrap_or_default(),
                "--motor" => cfg.motor = args.next().unwrap_or_default(),
                "--arm" => cfg.arm = args.next().unwrap_or_default(),
                "--port" => cfg.port = args.next().and_then(|s| s.parse().ok()).unwrap_or(8080),
                _ => {}
            }
        }
    }

    // ── Open hardware ──
    let mut camera = UsbCamera::open(&cfg.camera).expect("open camera");
    println!("Camera  : {}", cfg.camera);

    akars::pinmux::configure_for_device(&cfg.motor);
    let motor = Arc::new(Mutex::new(
        Motor::open(&MotorConfig { device: cfg.motor.clone(), ..Default::default() })
            .expect("open motor"),
    ));
    println!("Motor   : {}", cfg.motor);

    akars::pinmux::configure_for_device(&cfg.arm);
    let mut arm = Arm::open(&cfg.arm, 115200).expect("open arm");
    arm.restore_torque(0);
    thread::sleep(Duration::from_millis(50));
    arm.restore_torque(1);
    thread::sleep(Duration::from_millis(50));
    arm.restore_torque(2);
    thread::sleep(Duration::from_millis(50));
    arm.grab_pos();
    let arm = Arc::new(Mutex::new(arm));
    println!("Arm     : {}", cfg.arm);

    // ── Camera capture thread ──
    let camera_jpeg: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    {
        let frame_buf = camera_jpeg.clone();
        thread::Builder::new()
            .name("cam-stream".into())
            .spawn(move || {
                // Warm-up frames.
                for _ in 0..10 { let _ = camera.get_frame(); }
                loop {
                    match camera.get_frame() {
                        Ok(f) => {
                            match akars::image_bridge::yuv422p_to_jpeg_bytes(
                                &f.pixels, f.width as i32, f.height as i32, 75,
                            ) {
                                Ok(jpeg) => *frame_buf.lock().unwrap() = Some(jpeg),
                                Err(e) => eprintln!("[cam] JPEG: {e}"),
                            }
                        }
                        Err(e) => {
                            eprintln!("[cam] capture: {e}");
                            thread::sleep(Duration::from_millis(200));
                        }
                    }
                }
            })
            .expect("spawn cam-stream thread");
    }

    let state = Arc::new(AppState { camera_jpeg, motor, arm });

    // ── Build router ──
    let app = Router::new()
        .route("/", get(index))
        .route("/camera.jpg", get(camera_jpeg_handler))
        .route("/api/drive", post(drive))
        .route("/api/arm", post(arm_action))
        .with_state(state);

    // ── Start server ──
    let addr: SocketAddr = ([0, 0, 0, 0], cfg.port).into();
    println!("Listening on http://{}", addr);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });
}

// ── Handlers ──────────────────────────────────────────────────────────

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn camera_jpeg_handler(State(st): State<Arc<AppState>>) -> impl IntoResponse {
    match st.camera_jpeg.lock().unwrap().clone() {
        Some(data) => (
            [
                (header::CONTENT_TYPE, "image/jpeg"),
                (header::CACHE_CONTROL, "no-cache, no-store, must-revalidate"),
            ],
            data,
        ).into_response(),
        None => StatusCode::NO_CONTENT.into_response(),
    }
}

#[derive(Deserialize)]
struct DriveReq {
    action: String,
}

async fn drive(State(st): State<Arc<AppState>>, Json(req): Json<DriveReq>) -> impl IntoResponse {
    let mut m = st.motor.lock().unwrap();
    match req.action.as_str() {
        "forward"  => m.forward(50),
        "backward" => m.backward(30),
        "left"     => m.left(20),
        "right"    => m.right(20),
        "stop"     => m.standby(),
        _ => return (StatusCode::BAD_REQUEST, "unknown action").into_response(),
    }
    (StatusCode::OK, "ok").into_response()
}

#[derive(Deserialize)]
struct ArmReq {
    action: String,
}

async fn arm_action(State(st): State<Arc<AppState>>, Json(req): Json<ArmReq>) -> impl IntoResponse {
    let mut a = st.arm.lock().unwrap();
    match req.action.as_str() {
        "grab"      => a.grab(),
        "release"   => a.release(),
        "grab_pos"  => a.grab_pos(),
        _ => return (StatusCode::BAD_REQUEST, "unknown action").into_response(),
    }
    (StatusCode::OK, "ok").into_response()
}

// ── HTML page ─────────────────────────────────────────────────────────

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>CAM Web</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#eee;font-family:system-ui,sans-serif;text-align:center}
h2{margin:8px 0 4px;font-size:16px;color:#888}
#cam{width:100%;max-width:640px;border-radius:8px;background:#1a1a1a;margin-bottom:8px}
.pad{display:grid;grid-template:50px 50px 50px/50px 50px 50px;gap:6px;justify-content:center;margin:10px 0}
.pad button{border:none;border-radius:8px;background:#2a2a2a;color:#fff;font-size:20px;touch-action:manipulation}
.pad button:active{background:#555}
.arm-row{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin:10px 0}
.arm-row button{padding:10px 24px;border:none;border-radius:8px;font-size:15px;touch-action:manipulation}
.grab-btn{background:#2a7d2a;color:#fff}
.release-btn{background:#7d2a2a;color:#fff}
.ready-btn{background:#2a2a7d;color:#fff}
</style>
</head>
<body>
<h2>摄像头</h2>
<img id="cam" src="" alt="等待画面...">

<h2>底盘</h2>
<div class="pad">
  <span></span><button id="btn-fwd">&uarr;</button><span></span>
  <button id="btn-left">&larr;</button><button id="btn-stop">&#9632;</button><button id="btn-right">&rarr;</button>
  <span></span><button id="btn-bwd">&darr;</button><span></span>
</div>

<h2>手臂</h2>
<div class="arm-row">
  <button class="grab-btn" id="btn-grab">夹取</button>
  <button class="release-btn" id="btn-release">释放</button>
  <button class="ready-btn" id="btn-ready">就绪</button>
</div>

<script>
const send = (url, body) => fetch(url, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});

// ── Camera polling ──
(function poll(){let i=document.getElementById('cam');if(i){i.onload=()=>setTimeout(poll,200);i.onerror=()=>setTimeout(poll,800);i.src='/camera.jpg?t='+Date.now()}})();

// ── D-pad ──
['fwd','bwd','left','right'].forEach(d=>{
  const btn=document.getElementById('btn-'+d);
  btn.addEventListener('pointerdown',()=>send('/api/drive',{action:d}));
  btn.addEventListener('pointerup',()=>send('/api/drive',{action:'stop'}));
  btn.addEventListener('pointerleave',()=>send('/api/drive',{action:'stop'}));
});
document.getElementById('btn-stop').addEventListener('pointerdown',()=>send('/api/drive',{action:'stop'}));

// ── Arm ──
document.getElementById('btn-grab').addEventListener('click',()=>send('/api/arm',{action:'grab'}));
document.getElementById('btn-release').addEventListener('click',()=>send('/api/arm',{action:'release'}));
document.getElementById('btn-ready').addEventListener('click',()=>send('/api/arm',{action:'grab_pos'}));
</script>
</body>
</html>"#;
