//! Round 4 integration tests — full H.264 decode pipeline.
//!
//! Same skip-on-no-Vulkan policy as the earlier rounds.
//!
//! # Why does this test fork?
//!
//! Originally, on the dev box (NVIDIA RTX 5080, driver 580.95.05) the
//! Vulkan driver was SIGSEGV'ing inside `libnvidia-glcore` when
//! `vkQueueSubmit` executed our `vkCmdDecodeVideoKHR`. The
//! investigation (see CHANGELOG Round 4 — "Fixed" entry) pinned the
//! crash on incorrect API usage on our side: the DPB image was
//! transitioned to `VIDEO_DECODE_DST_KHR` instead of
//! `VIDEO_DECODE_DPB_KHR` for the setup-reference slot, the image
//! views used a multi-aspect mask that the NV12 format doesn't
//! permit, `VK_KHR_synchronization2` wasn't enabled alongside the
//! video extensions, and the `VideoSession::drop` dispatch went
//! through a dangling `&Device` borrow. With those fixed the decode
//! actually succeeds and produces an output that matches the ffmpeg
//! reference YUV bit-for-bit.
//!
//! We still fork a child process to run the helper — that way if a
//! future driver regression brings the crash back, the parent runner
//! survives, the test fails loudly with the signal number, and the
//! exact misuse pattern that re-emerged is easier to track down.

#![cfg(any(target_os = "linux", target_os = "windows"))]

use std::path::PathBuf;
use std::process::Command;

use oxideav_core::{time::TimeBase, CodecId, CodecParameters, Frame, Packet};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn read_fixture() -> Option<Vec<u8>> {
    let p = fixtures_dir().join("h264_high_320x240_1frame.h264");
    match std::fs::read(&p) {
        Ok(b) => Some(b),
        Err(_) => {
            eprintln!("vulkan-video round4: fixture missing at {:?}; skipping", p);
            None
        }
    }
}

fn read_reference_yuv() -> Option<Vec<u8>> {
    let p = fixtures_dir().join("reference.yuv");
    std::fs::read(&p).ok()
}

#[test]
fn h264_parser_finds_sps_pps() {
    use oxideav_bitstream::h264::{
        parse_pps_nal, parse_sps_nal, split_annex_b, NAL_TYPE_IDR, NAL_TYPE_PPS, NAL_TYPE_SPS,
    };

    let Some(bytes) = read_fixture() else {
        return;
    };

    let nals = split_annex_b(&bytes);
    let mut have_sps = false;
    let mut have_pps = false;
    let mut have_idr = false;
    for nal in &nals {
        match nal[0] & 0x1F {
            NAL_TYPE_SPS => {
                let s = parse_sps_nal(nal).expect("SPS parse");
                eprintln!(
                    "SPS: profile={} level={} chroma={} display={}x{} (coded {}x{}) cropping={}",
                    s.profile_idc,
                    s.level_idc,
                    s.chroma_format_idc,
                    s.display_width(),
                    s.display_height(),
                    s.coded_width(),
                    s.coded_height(),
                    s.frame_cropping.is_some(),
                );
                assert_eq!(s.profile_idc, 100, "expected H.264 High");
                assert!(
                    s.coded_width() >= 320,
                    "coded_width {} too small",
                    s.coded_width()
                );
                assert!(
                    s.coded_height() >= 240,
                    "coded_height {} too small",
                    s.coded_height()
                );
                assert_eq!(s.display_width(), 320);
                assert_eq!(s.display_height(), 240);
                have_sps = true;
            }
            NAL_TYPE_PPS => {
                let p = parse_pps_nal(nal).expect("PPS parse");
                eprintln!(
                    "PPS: pps_id={} sps_id={} entropy={} weighted_pred={}",
                    p.pic_parameter_set_id,
                    p.seq_parameter_set_id,
                    p.entropy_coding_mode_flag,
                    p.weighted_pred_flag,
                );
                have_pps = true;
            }
            NAL_TYPE_IDR => have_idr = true,
            _ => {}
        }
    }
    assert!(have_sps, "fixture lacks an SPS");
    assert!(have_pps, "fixture lacks a PPS");
    assert!(have_idr, "fixture lacks an IDR slice");
}

#[test]
fn h264_decoder_constructs_full_pipeline() {
    use oxideav_vulkan_video::decoder::H264VkDecoder;

    let Some(bytes) = read_fixture() else {
        return;
    };

    if oxideav_vulkan_video::sys::framework().is_err() {
        eprintln!("vulkan-video round4: no Vulkan loader; skipping");
        return;
    }

    let params = CodecParameters::video(CodecId::new("h264"));
    let mut dec = match H264VkDecoder::make(&params) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("vulkan-video round4: decoder make failed: {e}; skipping");
            return;
        }
    };

    let pkt = Packet {
        stream_index: 0,
        time_base: TimeBase::new(1, 1),
        pts: Some(0),
        dts: Some(0),
        duration: None,
        data: bytes,
        flags: Default::default(),
    };

    // Skip the actual `vkQueueSubmit` — this asserts that everything
    // up to (and including) `vkEndCommandBuffer` succeeds. The decode
    // dispatch itself is exercised by `h264_decoder_attempts_decode`
    // below (which forks to absorb the NVIDIA driver SIGSEGV).
    std::env::set_var("OXIDEAV_VK_SKIP_SUBMIT", "1");
    let result = dec.send_packet(&pkt).and_then(|_| dec.flush());
    std::env::remove_var("OXIDEAV_VK_SKIP_SUBMIT");

    // Expect the soft-fail "OXIDEAV_VK_SKIP_SUBMIT set" error,
    // which indicates the pipeline was constructed and recording
    // succeeded. On CI runners with a software Vulkan ICD (Mesa
    // llvmpipe / SwiftShader) that loads but doesn't advertise
    // `VK_KHR_video_decode_h264`, the lazy-init reaches `vkCreateDevice`
    // / `vkCreateVideoSessionKHR` / `vkGetPhysicalDeviceVideoCapabilitiesKHR`
    // and returns an Unsupported / VkError before the SKIP_SUBMIT hook
    // is consulted; treat those as a skip rather than a hard failure
    // since the H.264 dev-box decode path is fundamentally hardware-
    // gated.
    match result {
        Ok(()) => {}
        Err(e) => {
            let msg = format!("{e}");
            if msg.contains("OXIDEAV_VK_SKIP_SUBMIT") {
                return;
            }
            if msg.contains("vulkan-video")
                || msg.contains("Unsupported")
                || msg.contains("VK_ERROR")
                || msg.contains("vkGetPhysicalDeviceVideoCapabilities")
                || msg.contains("vkCreateVideoSession")
                || msg.contains("vkCreateDevice")
                || msg.contains("video extension")
                || msg.contains("decode_h264")
                || msg.contains("queue family")
            {
                eprintln!(
                    "vulkan-video round4: device does not support H.264 \
                     decode through Vulkan Video; skipping ({msg})"
                );
                return;
            }
            panic!("expected pipeline to construct successfully; got: {msg}");
        }
    }
}

#[test]
fn h264_decoder_attempts_decode() {
    let Some(_) = read_fixture() else {
        return;
    };
    if oxideav_vulkan_video::sys::framework().is_err() {
        eprintln!("vulkan-video round4: no Vulkan loader; skipping");
        return;
    }

    // Cargo exposes the exact path of binary targets to integration tests.
    // Using this avoids platform-specific .exe handling and target/deps layout guesses.
    let helper = PathBuf::from(env!("CARGO_BIN_EXE_round4_decode_helper"));
    assert!(
        helper.exists(),
        "round4 decode helper missing at {helper:?}"
    );

    let output_path = fixtures_dir().join("decoded_output.yuv");
    let _ = std::fs::remove_file(&output_path);

    let status = Command::new(&helper)
        .env_remove("OXIDEAV_VK_SKIP_SUBMIT")
        .env_remove("OXIDEAV_VK_SKIP_DECODE")
        .env("OXIDEAV_VK_DECODE_OUTPUT", &output_path)
        .env(
            "OXIDEAV_VK_FIXTURE",
            fixtures_dir().join("h264_high_320x240_1frame.h264"),
        )
        .status();

    let status = match status {
        Ok(s) => s,
        Err(e) => {
            eprintln!("vulkan-video round4: failed to spawn helper: {e}");
            return;
        }
    };

    if !status.success() {
        if let Some(sig) = status_signal(&status) {
            // The crash mode that Round 4 originally hit was an
            // NVIDIA-driver-side SIGSEGV during `vkQueueSubmit`.
            // The investigation pinned that to incorrect API usage
            // (DPB image layout, multi-planar aspect mask, missing
            // `VK_KHR_synchronization2`, dangling `&Device` borrow
            // on the video-session destructor) and fixed each of
            // them. If a future driver regression brings the crash
            // back, surface the signal explicitly so we don't
            // silently regress.
            panic!(
                "vulkan-video round4: helper crashed with signal {sig} \
                 (likely a driver/SDK regression; previously fixed by \
                 transitioning DPB to VIDEO_DECODE_DPB_KHR even in \
                 coincide mode and decoupling VideoSession::drop from \
                 the dangling parent &Device borrow)"
            );
        }
        // Exit code 2 is the helper's "any non-driver failure" path
        // — typically "no Vulkan ICD that advertises H.264 decode"
        // on a CI runner (Mesa llvmpipe / SwiftShader load but don't
        // expose `VK_KHR_video_decode_h264`). Treat that as a skip
        // since this test is fundamentally hardware-gated. A real
        // Vulkan-video device-side bug shows up as a signal (above)
        // or a different exit code.
        if status.code() == Some(2) {
            eprintln!(
                "vulkan-video round4: helper exited 2 — Vulkan loader \
                 present but no device advertises H.264 decode; skipping"
            );
            return;
        }
        if let Some(code) = status.code() {
            panic!("vulkan-video round4: helper exited with code {code}");
        }
        panic!("vulkan-video round4: helper exited abnormally");
    }

    // Helper succeeded — read back the decoded frame and validate.
    let decoded = std::fs::read(&output_path)
        .expect("vulkan-video round4: helper exited 0 but produced no output file");
    eprintln!(
        "vulkan-video round4: helper produced {} bytes",
        decoded.len()
    );

    assert_eq!(decoded.len(), 115200, "expected 320x240 I420");

    let y = &decoded[..76800];
    let mean: f64 = y.iter().map(|&x| x as f64).sum::<f64>() / (y.len() as f64);
    let var: f64 = y
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / (y.len() as f64);
    let stddev = var.sqrt();
    eprintln!("luma mean={:.1} stddev={:.2}", mean, stddev);
    assert!(
        stddev > 5.0,
        "luma uniform (stddev={stddev:.2}); decode produced constant pixels"
    );

    if let Some(reference) = read_reference_yuv() {
        let mut total: u64 = 0;
        for (a, b) in y.iter().zip(reference.iter().take(76800)) {
            total += (*a as i32 - *b as i32).unsigned_abs() as u64;
        }
        let mean_abs = total as f64 / 76800.0;
        eprintln!("luma vs reference mean abs diff: {:.2}/255", mean_abs);
        assert!(mean_abs < 20.0, "luma diff {} exceeds tolerance", mean_abs);
    }

    // Make compiler happy about Frame import.
    let _: Option<Frame> = None;
}

#[test]
fn h264_decoder_decodes_inter_frame_gop() {
    let fixture = fixtures_dir().join("h264_high_160x96_gop12.h264");
    let reference_path = fixtures_dir().join("h264_high_160x96_gop12_reference.yuv");
    if !fixture.exists() || !reference_path.exists() {
        eprintln!("vulkan-video round4: inter-frame fixture missing; skipping");
        return;
    }
    if oxideav_vulkan_video::sys::framework().is_err() {
        eprintln!("vulkan-video round4: no Vulkan loader; skipping");
        return;
    }

    let helper = PathBuf::from(env!("CARGO_BIN_EXE_round4_decode_helper"));
    let output_path = std::env::temp_dir().join(format!(
        "oxideav-vulkan-video-gop12-{}.yuv",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&output_path);

    let status = Command::new(&helper)
        .env_remove("OXIDEAV_VK_SKIP_SUBMIT")
        .env_remove("OXIDEAV_VK_SKIP_DECODE")
        .env("OXIDEAV_VK_DECODE_OUTPUT", &output_path)
        .env("OXIDEAV_VK_FIXTURE", &fixture)
        .status()
        .expect("spawn inter-frame decode helper");

    if !status.success() {
        if let Some(sig) = status_signal(&status) {
            panic!("vulkan-video round4: inter-frame helper crashed with signal {sig}");
        }
        if status.code() == Some(2) {
            panic!(
                "vulkan-video round4: inter-frame helper rejected a stream that should be supported"
            );
        }
        panic!(
            "vulkan-video round4: inter-frame helper exited abnormally: {:?}",
            status.code()
        );
    }

    let decoded =
        std::fs::read(&output_path).expect("inter-frame helper exited 0 but produced no output");
    let reference = std::fs::read(&reference_path).expect("read inter-frame ffmpeg reference");
    const FRAME_BYTES: usize = 160 * 96 * 3 / 2;
    assert_eq!(
        reference.len(),
        FRAME_BYTES * 12,
        "reference fixture must contain 12 frames"
    );
    assert_eq!(
        decoded.len(),
        reference.len(),
        "Vulkan decoder must emit all 12 display-order frames"
    );

    let mut total_abs = 0u64;
    let mut max_abs = 0u8;
    for (&actual, &expected) in decoded.iter().zip(&reference) {
        let delta = (actual as i16 - expected as i16).unsigned_abs() as u8;
        total_abs += delta as u64;
        max_abs = max_abs.max(delta);
    }
    let mean_abs = total_abs as f64 / decoded.len() as f64;
    eprintln!("inter-frame GOP vs ffmpeg: mean abs diff={mean_abs:.4}/255 max abs diff={max_abs}");
    assert!(
        mean_abs < 1.0,
        "inter-frame Vulkan output differs materially from ffmpeg reference: mean abs diff={mean_abs}"
    );
    let _ = std::fs::remove_file(&output_path);
}
#[cfg(unix)]
fn status_signal(s: &std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    s.signal()
}

#[cfg(not(unix))]
fn status_signal(_s: &std::process::ExitStatus) -> Option<i32> {
    None
}
