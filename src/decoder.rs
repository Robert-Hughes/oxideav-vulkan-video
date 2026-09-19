//! Vulkan Video H.264 decoder pipeline.
//!
//! The decoder is a packet-driven `oxideav_core::Decoder` implementation that
//! assembles Annex-B access units, delegates H.264 picture/POC/DPB semantics to
//! the shared `oxideav-h264` hardware frontend, lazily constructs the Vulkan
//! video session when SPS+PPS become available, then submits one
//! `vkCmdDecodeVideoKHR` per picture and copies the decoded NV12 image back into
//! a planar `VideoFrame`.
//!
//! # Pipeline shape
//!
//! 1. Parse complete Annex-B pictures through `H264PictureFrontend`, producing
//!    SPS/PPS, picture order counts, reference-picture state and stable DPB keys
//!    shared with the other hardware backends.
//! 2. Open a `VkInstance` (Vulkan 1.2), pick a discrete GPU that
//!    advertises `VK_KHR_video_decode_h264`, and create a `VkDevice`
//!    with a queue from a video-decode-capable queue family.
//! 3. Query H.264 decode capabilities to know the std-header version,
//!    DPB-slot upper bound, and bitstream alignment.
//! 4. Build a `VkVideoSessionKHR`, allocate + bind its memory backing.
//! 5. Build a `VkVideoSessionParametersKHR` carrying the parsed
//!    SPS + PPS.
//! 6. Allocate a 2D-array `VkImage` for the DPB (NV12 layout, layer
//!    count = `max_dpb_slots`), one or more output `VkImage`s sharing
//!    the same array layout for output-coincide implementations, and
//!    bind device memory to each.
//! 7. Allocate a host-visible `VkBuffer` for the H.264 bitstream
//!    payload (the full Annex-B picture, including SPS+PPS+slice).
//!    Note: NVIDIA's Vulkan driver accepts Annex-B start codes inside
//!    the bitstream payload — no MP4-AVCC reframing required.
//! 8. Allocate a host-visible staging `VkBuffer` sized for one NV12
//!    frame (luma + chroma plane) so we can read pixels back via
//!    `vkCmdCopyImageToBuffer`.
//! 9. Record + submit a single command buffer that:
//!    a. Transitions the DPB image to `VIDEO_DECODE_DPB_KHR` once and preserves it.
//!    b. Begins a coding scope with the live references plus reconstruction target bound.
//!    c. Issues the spec-mandated `RESET` control only for a new/reset sequence.
//!    d. Issues `vkCmdDecodeVideoKHR` with the picture's actual frame/POC metadata,
//!    setup slot and active reference slots mapped from the shared H.264 DPB.
//!    e. Ends coding scope.
//!    f. Transitions the output image to `TRANSFER_SRC_OPTIMAL` and
//!    `vkCmdCopyImageToBuffer` it to staging.
//! 10. Submit, wait via `vkQueueWaitIdle`, then memcpy from the
//!     mapped staging buffer into a planar `VideoFrame`.
//!
//! # Reality check
//!
//! Vulkan video decode remains driver-sensitive. The backend validates the
//! stream shape and advertised device capabilities before submission and
//! returns `Error::Unsupported` for codec tools it cannot translate safely,
//! allowing framework dispatch to choose another H.264 implementation rather
//! than approximating reference state. The GPU regression tests cover both the
//! original isolated-IDR case and a normal IDR/P/B GOP.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::ptr;

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, Result, VideoFrame, VideoPlane,
};
use oxideav_h264::access_unit::{starts_with_annex_b, AnnexBAccessUnitAssembler};
use oxideav_h264::dpb_output::{DpbOutput, OutputEntry};
use oxideav_h264::picture_frontend::{H264PictureFrontend, PreparedH264Picture};
use oxideav_h264::pps::Pps;
use oxideav_h264::ref_list::{DpbEntry, RefMarking};
use oxideav_h264::sps::Sps;

use crate::device::{Device, ExternalDevice};
use crate::instance::Instance;
use crate::physical_device::{
    PhysicalDevice, PhysicalDeviceType, VK_KHR_SYNCHRONIZATION_2_NAME,
    VK_KHR_VIDEO_DECODE_H264_NAME, VK_KHR_VIDEO_DECODE_QUEUE_NAME, VK_KHR_VIDEO_QUEUE_NAME,
};
use crate::sys::{
    self, StdVideoDecodeH264PictureInfo, StdVideoDecodeH264PictureInfoFlags,
    StdVideoDecodeH264ReferenceInfo, StdVideoH264PictureParameterSet, StdVideoH264PpsFlags,
    StdVideoH264SequenceParameterSet, StdVideoH264SpsFlags, VkBuffer, VkBufferCreateInfo,
    VkBufferImageCopy, VkCommandBuffer, VkCommandBufferAllocateInfo, VkCommandBufferBeginInfo,
    VkCommandPool, VkCommandPoolCreateInfo, VkComponentMapping, VkDeviceMemory, VkExtent2D,
    VkExtent3D, VkImage, VkImageCreateInfo, VkImageMemoryBarrier, VkImageSubresourceLayers,
    VkImageSubresourceRange, VkImageView, VkImageViewCreateInfo, VkMemoryAllocateInfo,
    VkMemoryRequirements, VkOffset2D, VkOffset3D, VkPhysicalDeviceMemoryProperties, VkSubmitInfo,
    VkVideoBeginCodingInfoKHR, VkVideoCodingControlInfoKHR, VkVideoDecodeH264DpbSlotInfoKHR,
    VkVideoDecodeH264PictureInfoKHR, VkVideoDecodeH264ProfileInfoKHR,
    VkVideoDecodeH264SessionParametersAddInfoKHR, VkVideoDecodeH264SessionParametersCreateInfoKHR,
    VkVideoDecodeInfoKHR, VkVideoEndCodingInfoKHR, VkVideoPictureResourceInfoKHR,
    VkVideoProfileInfoKHR, VkVideoProfileListInfoKHR, VkVideoReferenceSlotInfoKHR,
    VkVideoSessionParametersCreateInfoKHR, VkVideoSessionParametersKHR, VK_ACCESS_MEMORY_READ_BIT,
    VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_API_VERSION_1_2,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR,
    VK_COMMAND_BUFFER_LEVEL_PRIMARY, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
    VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_ASPECT_PLANE_1_BIT,
    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR, VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR,
    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_TYPE_2D, VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR, VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR,
    VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_QUEUE_FAMILY_IGNORED, VK_SAMPLE_COUNT_1_BIT,
    VK_SHARING_MODE_EXCLUSIVE, VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, VK_STRUCTURE_TYPE_SUBMIT_INFO,
    VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR, VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR, VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR, VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR, VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR, VK_SUCCESS,
    VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR, VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
    VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR, VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
    VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR,
};
use crate::video::{query_video_decode_h264_capabilities, VideoSession};

// ─────────────────────────── helpers ─────────────────────────────────────────

fn vk_err(op: &'static str, r: i32) -> Error {
    Error::other(format!("{op} returned VkResult({r})"))
}

/// Round `v` up to the nearest multiple of `align`. Caller ensures
/// `align != 0`.
fn align_up(v: u64, align: u64) -> u64 {
    if align == 0 {
        v
    } else {
        v.div_ceil(align) * align
    }
}

fn bitstream_buffer_capacity(coded_width: u32, coded_height: u32, alignment: u64) -> u64 {
    let coded_pixels = u64::from(coded_width).saturating_mul(u64::from(coded_height));
    let picture_budget = coded_pixels.saturating_mul(2).max(4 * 1024 * 1024);
    align_up(picture_budget, alignment.max(1))
}

fn output_dimensions(
    display_width: u32,
    display_height: u32,
    coded_width: u32,
    coded_height: u32,
) -> (u32, u32, u32) {
    let width = display_width.min(coded_width);
    let height = display_height.min(coded_height);
    (width, height, height.div_ceil(2))
}

/// Walk the Annex-B bitstream and return the byte offsets (relative
/// to the start of `bitstream`) of each VCL slice's start-code prefix.
///
/// VCL slices are NAL types 1 (non-IDR slice) and 5 (IDR slice). The
/// emitted offset points to the FIRST byte of the start-code prefix
/// (0x000001 or 0x00000001).
fn compute_slice_offsets(bitstream: &[u8]) -> Vec<u32> {
    let mut offsets = Vec::new();
    let mut pos = 0usize;
    let len = bitstream.len();

    while pos + 4 <= len {
        let sc_len = if bitstream[pos] == 0
            && bitstream[pos + 1] == 0
            && bitstream[pos + 2] == 0
            && bitstream[pos + 3] == 1
        {
            4
        } else if bitstream[pos] == 0 && bitstream[pos + 1] == 0 && bitstream[pos + 2] == 1 {
            3
        } else {
            pos += 1;
            continue;
        };
        let nal_byte_pos = pos + sc_len;
        if nal_byte_pos < len {
            let nt = bitstream[nal_byte_pos] & 0x1F;
            if nt == 1 || nt == 5 {
                offsets.push(pos as u32);
            }
        }
        pos += sc_len;
    }
    offsets
}

/// Whether `pd` would be surfaced by [`crate::engine_info`]. The
/// decoder uses the same predicate so `device_index` indexes into the
/// same filtered list a CLI consumer of `engine_info()` sees.
///
/// Mirrors `crate::engine::build_device_info`'s admit rule:
/// `device_type ∈ {Discrete, Integrated, Virtual} GPU OR any
/// VK_KHR_video_* extension advertised`. CPU / Other ICDs without
/// any video extension are skipped — the same set engine_info()
/// would have skipped.
fn engine_info_filter_admits(pd: &PhysicalDevice<'_>) -> bool {
    let props = pd.properties();
    let useful_type = matches!(
        props.device_type,
        PhysicalDeviceType::DiscreteGpu
            | PhysicalDeviceType::IntegratedGpu
            | PhysicalDeviceType::VirtualGpu
    );
    if useful_type {
        return true;
    }
    let v = pd.supports_video_extensions();
    v.queue_khr || v.decode_h264 || v.decode_h265 || v.decode_av1 || v.encode_h264 || v.encode_h265
}

/// Pick the first memory type whose bit is set in `type_bits` AND
/// whose flags satisfy `required`.
fn pick_memory_type(
    props: &VkPhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: u32,
) -> Option<u32> {
    for i in 0..props.memory_type_count {
        if type_bits & (1u32 << i) == 0 {
            continue;
        }
        if (props.memory_types[i as usize].property_flags & required) == required {
            return Some(i);
        }
    }
    None
}

/// Wrapper that owns the Vulkan objects backing one decode session.
///
/// Drop tears them down in the order the spec requires: command pool +
/// fence first, then images / image-views / buffers, then session
/// parameters, then session, then memory, then device, then instance.
/// Per-instance capability data captured at construction time so we
/// don't need to re-query during every decode_picture call.
#[allow(dead_code)]
struct CachedCaps {
    bitstream_offset_alignment: u64,
    bitstream_size_alignment: u64,
}

struct DecoderState {
    // ─── Drop order matters! ──────────────────────────────────────
    // Field declaration order = drop order. The Vulkan spec requires
    // children to be destroyed before their parents:
    //   command_buffer → command_pool → buffers/images/views/memory
    //   → session_params → session → device → instance.
    // Putting children FIRST and parents LAST gives us that order.
    // We do explicit destruction in `Drop` (because most of the
    // Vulkan handles are non-RAII raw types) but we still rely on
    // `device` and `instance` being kept alive until the very end.
    /// Pre-recorded command buffer for one decode dispatch. Freed
    /// before `command_pool` in Drop.
    command_buffer: VkCommandBuffer,
    command_pool: VkCommandPool,

    bitstream_buffer: VkBuffer,
    bitstream_memory: VkDeviceMemory,
    bitstream_size: u64,

    staging_buffer: VkBuffer,
    staging_memory: VkDeviceMemory,
    staging_size: u64,

    output_image_view: VkImageView,
    output_image: VkImage,
    output_memory: VkDeviceMemory,

    dpb_image_view: VkImageView,
    dpb_image: VkImage,
    dpb_memory: VkDeviceMemory,

    session_params: VkVideoSessionParametersKHR,
    /// Owns `vkDestroyVideoSessionKHR`. Must drop before `device`.
    session: Option<VideoSession<'static>>,

    queue_family_index: u32,
    /// Which queue within `queue_family_index` submissions go to.
    /// Always `0` for the self-created path; imported devices
    /// ([`ExternalDevice::queue_index`]) may pick another queue.
    queue_index: u32,
    /// Held for diagnostics / future round expansion.
    #[allow(dead_code)]
    physical_device_handle: sys::VkPhysicalDevice,

    /// Owns `vkDestroyDevice`. Must drop before `instance`.
    device: Device,

    /// Owns `vkDestroyInstance`. Drops last.
    #[allow(dead_code)]
    instance: Instance,

    width: u32,
    height: u32,
    /// Luma plane row pitch in *bytes* (also = texels, plane 0 is
    /// 1 byte/texel).
    luma_stride: u32,
    /// Chroma plane row pitch in *texels* of plane 1 (R8G8 = 2
    /// bytes/texel). For NV12 a packed row is `width/2` chroma
    /// texels = `width` bytes. Vulkan's
    /// `VkBufferImageCopy::buffer_row_length` is measured in texels of
    /// the source plane, not bytes — using `width` here would tell
    /// the driver to skip 2× the real bytes/row and overrun the
    /// staging buffer (VUID-vkCmdCopyImageToBuffer-pRegions-00183).
    chroma_stride: u32,
    chroma_height: u32,

    /// Whether DPB and output coincide on this driver. NVIDIA on
    /// Linux reports `DPB_AND_OUTPUT_COINCIDE_BIT_KHR` and we use the
    /// same array image for both. AMD historically reports `DISTINCT`,
    /// in which case `output_image` is its own image.
    coincide: bool,
    /// Number of DPB layers/slots allocated for this video session.
    dpb_slot_count: u32,
    /// The DPB array is transitioned from UNDEFINED to DPB layout once, then
    /// kept there for the lifetime of the session except for the current
    /// coincide-mode layer while it is copied to staging.
    dpb_initialized: bool,
    /// Distinct output images likewise leave UNDEFINED only once.
    output_initialized: bool,
    /// Vulkan coding state must be RESET on the first decode submission.
    needs_reset: bool,

    /// Bitstream buffer offset alignment from caps. Currently unused
    /// (we use the size alignment for the decode srcBufferRange) but
    /// kept for future round expansion (multi-slice / multi-picture
    /// streams will need explicit offset alignment).
    #[allow(dead_code)]
    bitstream_offset_alignment: u64,
    /// Bitstream buffer size alignment from caps.
    bitstream_size_alignment: u64,
}

// SAFETY: every Vulkan handle in `DecoderState` is externally
// synchronised — we only ever drive them from `&mut self` so there
// are no aliased mutations from other threads. The Device / Instance
// already encapsulate the same single-threaded contract.
unsafe impl Send for DecoderState {}

impl Drop for DecoderState {
    fn drop(&mut self) {
        // SAFETY: we hold an exclusive `&mut self` for the duration of
        // Drop; the device handle is still alive (Device's Drop runs
        // after this on field declaration order).
        let dfns: &crate::device::DeviceFns = self.device.fns();

        // Wait for *every* queue, not just our video queue, so that
        // any host-visible memory copy issued via TRANSFER on the
        // same queue family (or any other family if a future round
        // splits it) is fully retired before we tear handles down.
        // NVIDIA's session destructor is sensitive to in-flight
        // command buffers referencing the bound DPB.
        unsafe {
            (dfns.queue_wait_idle)(
                self.device
                    .queue_indexed(self.queue_family_index, self.queue_index)
                    .handle(),
            );
        }

        // Tear down in spec order:
        //   command buffers → command pool → session params →
        //   session → DPB / output images + their views → buffers →
        //   bound memory.
        //
        // The video session is destroyed BEFORE any image whose
        // memory backed a DPB / output picture-resource bound on it.
        // NVIDIA's `vkDestroyVideoSessionKHR` walks the most-
        // recently-used DPB binding and dereferences the image's
        // backing memory; if we've already freed the image (and its
        // memory) it crashes inside the session destructor —
        // exactly the SIGSEGV the gdb backtrace points at.
        // Destroy session_params then session FIRST.
        unsafe {
            if !self.command_buffer.is_null() && !self.command_pool.is_null() {
                (dfns.free_command_buffers)(
                    self.device.handle(),
                    self.command_pool,
                    1,
                    &self.command_buffer,
                );
                self.command_buffer = ptr::null_mut();
            }
            if !self.command_pool.is_null() {
                (dfns.destroy_command_pool)(self.device.handle(), self.command_pool, ptr::null());
                self.command_pool = ptr::null_mut();
            }
            if !self.session_params.is_null() {
                (dfns.destroy_video_session_parameters_khr)(
                    self.device.handle(),
                    self.session_params,
                    ptr::null(),
                );
                self.session_params = ptr::null_mut();
            }
        }

        // Tear down the video session via OUR `&self.device`, not
        // via the `&Device` borrow embedded inside `VideoSession`.
        // The latter was hijacked-to-`'static` during construction
        // and points to the *original* stack address of the local
        // `device` binding before it was moved into
        // `DecoderState::device` — that pointer is dangling by now,
        // and dispatching through `(self.session.device.fns()).destroy_video_session_khr`
        // dereferences garbage. The detach API below transfers
        // ownership of the handle and bound-memory list to us so the
        // session's own `Drop` becomes a no-op.
        if let Some(mut session) = self.session.take() {
            let (session_handle, bound_memory) = session.detach();
            unsafe {
                if !session_handle.is_null() {
                    (dfns.destroy_video_session_khr)(
                        self.device.handle(),
                        session_handle,
                        ptr::null(),
                    );
                }
                for m in bound_memory {
                    if !m.is_null() {
                        (dfns.free_memory)(self.device.handle(), m, ptr::null());
                    }
                }
            }
            // Drop the now-empty VideoSession — its Drop sees
            // `handle.is_null()` and `bound_memory.is_empty()` and
            // returns without touching the (dangling) device borrow.
            drop(session);
        }

        unsafe {
            if !self.bitstream_buffer.is_null() {
                (dfns.destroy_buffer)(self.device.handle(), self.bitstream_buffer, ptr::null());
                self.bitstream_buffer = ptr::null_mut();
            }
            if !self.bitstream_memory.is_null() {
                (dfns.free_memory)(self.device.handle(), self.bitstream_memory, ptr::null());
                self.bitstream_memory = ptr::null_mut();
            }
            if !self.staging_buffer.is_null() {
                (dfns.destroy_buffer)(self.device.handle(), self.staging_buffer, ptr::null());
                self.staging_buffer = ptr::null_mut();
            }
            if !self.staging_memory.is_null() {
                (dfns.free_memory)(self.device.handle(), self.staging_memory, ptr::null());
                self.staging_memory = ptr::null_mut();
            }
            if !self.output_image_view.is_null() {
                (dfns.destroy_image_view)(
                    self.device.handle(),
                    self.output_image_view,
                    ptr::null(),
                );
                self.output_image_view = ptr::null_mut();
            }
            if !self.output_image.is_null() && self.output_image != self.dpb_image {
                (dfns.destroy_image)(self.device.handle(), self.output_image, ptr::null());
            }
            self.output_image = ptr::null_mut();
            if !self.output_memory.is_null() && self.output_memory != self.dpb_memory {
                (dfns.free_memory)(self.device.handle(), self.output_memory, ptr::null());
            }
            self.output_memory = ptr::null_mut();
            if !self.dpb_image_view.is_null() {
                (dfns.destroy_image_view)(self.device.handle(), self.dpb_image_view, ptr::null());
                self.dpb_image_view = ptr::null_mut();
            }
            if !self.dpb_image.is_null() {
                (dfns.destroy_image)(self.device.handle(), self.dpb_image, ptr::null());
                self.dpb_image = ptr::null_mut();
            }
            if !self.dpb_memory.is_null() {
                (dfns.free_memory)(self.device.handle(), self.dpb_memory, ptr::null());
                self.dpb_memory = ptr::null_mut();
            }
        }
    }
}

// ─────────────────────────── public decoder ──────────────────────────────────

/// Vulkan Video H.264 streaming decoder.
///
/// Parsing, POC derivation and decoded-reference-picture marking are delegated
/// to oxideav-h264's shared hardware frontend. This backend owns the Vulkan
/// session/resources and the mapping from opaque H.264 DPB keys to Vulkan slots.
pub struct H264VkDecoder {
    codec_id: CodecId,
    au_assembler: AnnexBAccessUnitAssembler,
    frontend: H264PictureFrontend,
    state: Option<DecoderState>,
    dpb_slots: HashMap<u32, u32>,
    /// DPB slots that Vulkan still considers active, including stale slots
    /// whose H.264 keys have already left the frontend DPB.
    active_slots: HashSet<u32>,
    output_dpb: DpbOutput<VideoFrame>,
    ready: VecDeque<VideoFrame>,
    eof: bool,
    device_index: u32,
    external: Option<ExternalDevice>,
}

impl H264VkDecoder {
    fn new(device_index: u32, external: Option<ExternalDevice>) -> Self {
        Self {
            codec_id: CodecId::new("h264"),
            au_assembler: AnnexBAccessUnitAssembler::default(),
            frontend: H264PictureFrontend::new(),
            state: None,
            dpb_slots: HashMap::new(),
            active_slots: HashSet::new(),
            output_dpb: DpbOutput::new(4, 4),
            ready: VecDeque::new(),
            eof: false,
            device_index,
            external,
        }
    }

    /// Factory used by the framework registry.
    pub fn make(params: &CodecParameters) -> Result<Box<dyn oxideav_core::Decoder>> {
        sys::vtable().map_err(|e| Error::unsupported(format!("vulkan-video: {e}")))?;
        if !params.extradata.is_empty() && !starts_with_annex_b(&params.extradata) {
            return Err(Error::unsupported(
                "vulkan-video: H.264 streaming decoder currently supports Annex-B input only",
            ));
        }
        Ok(Box::new(Self::new(params.device_index.unwrap_or(0), None)))
    }

    /// Construct a decoder that runs on an application-owned Vulkan device.
    ///
    /// # Safety
    ///
    /// Every handle in external must remain valid and mutually consistent until
    /// the returned decoder is dropped. The selected queue must not be used
    /// concurrently while the decoder is submitting work.
    pub unsafe fn make_with_device(
        params: &CodecParameters,
        external: ExternalDevice,
    ) -> Result<Box<dyn oxideav_core::Decoder>> {
        if !params.extradata.is_empty() && !starts_with_annex_b(&params.extradata) {
            return Err(Error::unsupported(
                "vulkan-video: H.264 streaming decoder currently supports Annex-B input only",
            ));
        }
        Ok(Box::new(Self::new(0, Some(external))))
    }

    fn ensure_state(&mut self, sps: &Sps, pps: &Pps) -> Result<()> {
        if self.state.is_some() {
            return Ok(());
        }
        let state = match &self.external {
            Some(ext) => DecoderState::create_external(sps, pps, ext)?,
            None => DecoderState::create(sps, pps, self.device_index)?,
        };
        self.state = Some(state);
        Ok(())
    }

    fn ensure_output_dpb(&mut self, sps: &Sps) {
        let explicit_reorder = sps
            .vui
            .as_ref()
            .and_then(|v| v.bitstream_restriction.as_ref())
            .map(|br| br.max_num_reorder_frames)
            .unwrap_or(0);
        let explicit_buffering = sps
            .vui
            .as_ref()
            .and_then(|v| v.bitstream_restriction.as_ref())
            .map(|br| br.max_dec_frame_buffering)
            .unwrap_or(0);
        let reorder = sps.max_num_ref_frames.max(explicit_reorder).clamp(4, 16);
        let buffering = sps
            .max_num_ref_frames
            .max(explicit_buffering)
            .max(reorder)
            .min(16);
        if self.output_dpb.max_num_reorder_frames == reorder
            && self.output_dpb.max_dec_frame_buffering == buffering
        {
            return;
        }
        self.drain_output_dpb();
        self.output_dpb = DpbOutput::new(reorder, buffering);
    }

    fn drain_output_dpb(&mut self) {
        let pending = self.output_dpb.flush();
        self.ready.extend(pending.into_iter().map(|e| e.picture));
    }

    fn target_slot(&self, picture: &PreparedH264Picture) -> Result<u32> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| Error::other("vulkan-video: decoder state is not initialised"))?;

        for reference in &picture.references {
            if reference.marking != RefMarking::Unused
                && !self.dpb_slots.contains_key(&reference.dpb_key)
            {
                return Err(Error::other(format!(
                    "vulkan-video: live H.264 DPB key {} has no Vulkan slot",
                    reference.dpb_key
                )));
            }
        }

        for slot in 0..state.dpb_slot_count {
            let occupied = if picture.reference_reset {
                false
            } else {
                self.dpb_slots.values().any(|&mapped| mapped == slot)
            };
            if !occupied {
                return Ok(slot);
            }
        }
        Err(Error::unsupported(
            "vulkan-video: no free DPB slot is available for the current H.264 picture",
        ))
    }

    fn decode_access_unit(&mut self, packet: &Packet) -> Result<()> {
        if packet.data.is_empty() {
            return Ok(());
        }
        if !starts_with_annex_b(&packet.data) {
            return Err(Error::unsupported(
                "vulkan-video: H.264 streaming decoder received non-Annex-B access unit",
            ));
        }

        let Some(picture) = self.frontend.prepare_access_unit(&packet.data)? else {
            return Ok(());
        };
        validate_picture_shape(&picture)?;
        if !picture.synthetic_references.is_empty() {
            return Err(Error::unsupported(
                "vulkan-video: non-existing H.264 frame_num-gap references are not yet materialised",
            ));
        }

        if picture.reference_reset {
            let pending = self.output_dpb.flush();
            if !picture.no_output_of_prior_pics {
                self.ready.extend(pending.into_iter().map(|e| e.picture));
            }
        }

        self.ensure_output_dpb(&picture.sps);
        self.ensure_state(&picture.sps, &picture.pps)?;
        let target_slot = self.target_slot(&picture)?;
        let target_was_active = self.active_slots.contains(&target_slot);
        let reset_session =
            picture.reference_reset || self.state.as_ref().is_some_and(|state| state.needs_reset);

        let mut frame = self
            .state
            .as_mut()
            .expect("state ensured above")
            .decode_picture(
                &packet.data,
                &picture,
                target_slot,
                target_was_active,
                &self.dpb_slots,
                reset_session,
            )?;
        frame.pts = packet.pts;

        let frame_num = picture.header.frame_num;
        let pic_order_cnt = picture.poc.pic_order_cnt;
        let reference_reset = picture.reference_reset;
        let is_reference = picture.is_reference();
        let commit = self.frontend.commit(picture);

        if reset_session {
            self.active_slots.clear();
        }
        if reference_reset {
            self.dpb_slots.clear();
        }
        for key in commit.dead_dpb_keys {
            self.dpb_slots.remove(&key);
        }
        if let Some(key) = commit.current_dpb_key {
            self.dpb_slots.insert(key, target_slot);
        }
        if is_reference {
            self.active_slots.insert(target_slot);
        } else {
            self.active_slots.remove(&target_slot);
        }

        if commit.mmco5 {
            let pending = self.output_dpb.reset();
            self.ready.extend(pending.into_iter().map(|e| e.picture));
        }

        if let Some(bumped) = self.output_dpb.push(OutputEntry {
            picture: frame,
            pic_order_cnt,
            frame_num,
            needed_for_output: false,
        }) {
            self.ready.push_back(bumped.picture);
        }
        Ok(())
    }

    fn clear_stream_state(&mut self) {
        self.au_assembler.reset();
        self.frontend.reset();
        self.dpb_slots.clear();
        self.output_dpb = DpbOutput::new(4, 4);
        self.ready.clear();
        self.eof = false;
        self.state = None;
    }
}

unsafe impl Send for H264VkDecoder {}

impl oxideav_core::Decoder for H264VkDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.eof {
            return Err(Error::invalid(
                "vulkan-video: H.264 decoder received packet after flush",
            ));
        }
        let completed = self.au_assembler.push(packet)?;
        for access_unit in completed {
            self.decode_access_unit(&access_unit)?;
        }
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(frame) = self.ready.pop_front() {
            return Ok(Frame::Video(frame));
        }
        if self.eof && self.output_dpb.is_empty() {
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        if !self.eof {
            if let Some(access_unit) = self.au_assembler.flush() {
                self.decode_access_unit(&access_unit)?;
            }
            self.drain_output_dpb();
            self.eof = true;
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.clear_stream_state();
        Ok(())
    }
}

// ─────────────────────────── DecoderState — heavy lifting ────────────────────

impl DecoderState {
    fn create(sps: &Sps, pps: &Pps, device_index: u32) -> Result<Self> {
        // ── Instance ────────────────────────────────────────────
        let instance = Instance::new("oxideav-vulkan-video", VK_API_VERSION_1_2)
            .map_err(|e| Error::unsupported(format!("vulkan-video: {e}")))?;

        // ── Pick a video-decode-capable physical device ─────────
        //
        // `device_index` is interpreted against the *same* filter
        // `engine_info()` applies (see `crate::engine`): every
        // physical device whose type is Discrete/Integrated/Virtual
        // GPU OR that advertises at least one `VK_KHR_video_*`
        // extension is included, in `Instance::physical_devices()`
        // enumeration order. The two MUST stay in sync — an
        // `engine_info()` consumer that prints "device 1" and then
        // passes `with_device_index(1)` expects the decoder to bind
        // to that exact same physical device.
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: enumerating physical devices");
        }
        let devices = instance
            .physical_devices()
            .map_err(|e| Error::unsupported(format!("vulkan-video: {e}")))?;
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: {} physical devices found", devices.len());
        }

        // Filter to the same set engine_info() exposes.
        let filtered_indices: Vec<usize> = devices
            .iter()
            .enumerate()
            .filter(|(_, d)| engine_info_filter_admits(d))
            .map(|(i, _)| i)
            .collect();
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: {} physical device(s) survive engine_info filter",
                filtered_indices.len(),
            );
        }
        if (device_index as usize) >= filtered_indices.len() {
            return Err(Error::unsupported(format!(
                "vulkan-video: device_index {device_index} out of range (0..{})",
                filtered_indices.len()
            )));
        }
        let raw_idx = filtered_indices[device_index as usize];
        let chosen_dev = &devices[raw_idx];
        let support = chosen_dev.supports_video_extensions();
        if !support.queue_khr || !support.decode_h264 {
            return Err(Error::unsupported(format!(
                "vulkan-video: device_index {device_index} does not support H.264 decode \
                 (queue_khr={} decode_h264={})",
                support.queue_khr, support.decode_h264
            )));
        }
        let qfi = chosen_dev
            .video_queue_family_indices()
            .first()
            .copied()
            .ok_or_else(|| {
                Error::unsupported(format!(
                    "vulkan-video: device_index {device_index} advertises decode_h264 \
                     but reports no video-capable queue family"
                ))
            })?;
        let pd_handle = chosen_dev.handle();
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: chosen filtered device_index={} (raw {}) with qfi={}",
                device_index, raw_idx, qfi,
            );
        }
        // Re-enumerate so we don't keep the borrow on `instance` while
        // using `instance` later.
        drop(devices);

        // ── Device ──────────────────────────────────────────────
        // We re-enumerate one more time and create the Device.
        let pds = instance
            .physical_devices()
            .map_err(|e| Error::unsupported(format!("vulkan-video: physical_devices2: {e}")))?;
        let pd = pds
            .iter()
            .find(|p| p.handle() == pd_handle)
            .ok_or_else(|| Error::other("vulkan-video: pd lookup2 failed"))?;
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: creating device");
        }
        let device = Device::new(
            pd,
            qfi,
            &[
                // VK_KHR_video_queue is specified in terms of sync2
                // stage / access bits, so the loader's validation
                // requires sync2 to be enabled alongside it
                // (VUID-vkCreateDevice-ppEnabledExtensionNames-01387).
                // sync2 is core in Vulkan 1.3 but Round 2+ requests
                // 1.2 explicitly, so we list it by name here.
                VK_KHR_SYNCHRONIZATION_2_NAME,
                VK_KHR_VIDEO_QUEUE_NAME,
                VK_KHR_VIDEO_DECODE_QUEUE_NAME,
                VK_KHR_VIDEO_DECODE_H264_NAME,
            ],
        )
        .map_err(|e| Error::unsupported(format!("vulkan-video: vkCreateDevice: {e}")))?;
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: device OK");
        }
        drop(pds);

        Self::build(sps, pps, instance, pd_handle, qfi, 0, device)
    }

    /// Construct the decode pipeline on an application-supplied
    /// device ([`ExternalDevice`]) instead of creating our own
    /// instance / device (GitHub issue #2).
    ///
    /// The imported handles are wrapped non-owning — nothing in
    /// `Drop` destroys the caller's instance or device; only the
    /// objects this crate created on top of them (session, images,
    /// buffers, command pool, …) are torn down.
    ///
    /// # Safety contract (discharged by `make_with_device`)
    ///
    /// The `ExternalDevice` handles were vouched for by the caller of
    /// [`H264VkDecoder::make_with_device`]; see the contract there.
    fn create_external(sps: &Sps, pps: &Pps, ext: &ExternalDevice) -> Result<Self> {
        // SAFETY: handle validity / lifetime / synchronisation were
        // guaranteed by the unsafe `make_with_device` caller.
        let (instance, device) = unsafe { ext.import() }
            .map_err(|e| Error::unsupported(format!("vulkan-video: import device: {e}")))?;

        // Cheap sanity checks with clear diagnostics before we get
        // deep into session construction.
        {
            // SAFETY: same caller contract — the physical device
            // belongs to the imported instance.
            let pd = unsafe { instance.physical_device_from_raw(ext.physical_device) };
            let support = pd.supports_video_extensions();
            if !support.queue_khr || !support.decode_h264 {
                return Err(Error::unsupported(format!(
                    "vulkan-video: imported device does not support H.264 decode \
                     (queue_khr={} decode_h264={})",
                    support.queue_khr, support.decode_h264
                )));
            }
            if !pd
                .video_queue_family_indices()
                .contains(&ext.queue_family_index)
            {
                return Err(Error::unsupported(format!(
                    "vulkan-video: imported queue_family_index {} is not video-capable",
                    ext.queue_family_index
                )));
            }
        }

        Self::build(
            sps,
            pps,
            instance,
            ext.physical_device,
            ext.queue_family_index,
            ext.queue_index,
            device,
        )
    }

    /// Shared tail of [`DecoderState::create`] /
    /// [`DecoderState::create_external`]: everything downstream of
    /// having an `Instance`, a chosen physical device + video queue
    /// family, and a `VkDevice` (owned or imported) — capability
    /// query, video session + parameters, DPB / output images,
    /// bitstream + staging buffers, command pool.
    fn build(
        sps: &Sps,
        pps: &Pps,
        instance: Instance,
        pd_handle: sys::VkPhysicalDevice,
        qfi: u32,
        queue_index: u32,
        device: Device,
    ) -> Result<Self> {
        // ── Capabilities ────────────────────────────────────────
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: querying caps");
        }
        let pds = instance
            .physical_devices()
            .map_err(|e| Error::unsupported(format!("vulkan-video: physical_devices: {e}")))?;
        let pd = pds
            .iter()
            .find(|p| p.handle() == pd_handle)
            .ok_or_else(|| Error::other("vulkan-video: physical device disappeared"))?;
        let profile_idc = h264_profile_idc(sps.profile_idc)?;
        let caps = query_video_decode_h264_capabilities(pd, profile_idc)
            .map_err(|e| Error::unsupported(format!("vulkan-video: caps: {e}")))?;
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: caps OK ({}x{} dpb={} bsalign={})",
                caps.max_coded_extent.0,
                caps.max_coded_extent.1,
                caps.max_dpb_slots,
                caps.min_bitstream_buffer_offset_alignment
            );
        }

        // Driver capability flag: DPB & output coincide?
        let coincide =
            caps.decode_capability_flags & 0x1 != 0 // VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR
                ;

        // Memory properties for picking memory types.
        let mem_props = {
            let mut p: VkPhysicalDeviceMemoryProperties = unsafe { std::mem::zeroed() };
            unsafe {
                (pd.instance_fns().get_physical_device_memory_properties)(pd.handle(), &mut p);
            }
            p
        };
        drop(pds);

        // ── Video session ───────────────────────────────────────
        // We need the lifetime of session and the underlying device
        // to align; we hide this with `transmute` below into an owned
        // `VideoSession<'static>` whose Drop runs before the Device's
        // Drop because of struct field ordering.
        let pds = instance
            .physical_devices()
            .map_err(|e| Error::unsupported(format!("vulkan-video: physical_devices3: {e}")))?;
        let pd = pds
            .iter()
            .find(|p| p.handle() == pd_handle)
            .ok_or_else(|| Error::other("vulkan-video: pd lookup3 failed"))?;
        let (coded_width, coded_height) = coded_dimensions(sps);
        if coded_width > caps.max_coded_extent.0 || coded_height > caps.max_coded_extent.1 {
            return Err(Error::unsupported(format!(
                "vulkan-video: H.264 coded extent {}x{} exceeds device limit {}x{}",
                coded_width, coded_height, caps.max_coded_extent.0, caps.max_coded_extent.1
            )));
        }
        if sps.max_num_ref_frames > caps.max_active_reference_pictures {
            return Err(Error::unsupported(format!(
                "vulkan-video: H.264 stream requires {} references but device supports {}",
                sps.max_num_ref_frames, caps.max_active_reference_pictures
            )));
        }
        let max_w = coded_width.max(caps.min_coded_extent.0).max(16);
        let max_h = coded_height.max(caps.min_coded_extent.1).max(16);
        let dpb_slots = caps.max_dpb_slots.clamp(1, 17);
        if dpb_slots <= sps.max_num_ref_frames {
            return Err(Error::unsupported(format!(
                "vulkan-video: {} DPB slots cannot hold {} references plus a decode target",
                dpb_slots, sps.max_num_ref_frames
            )));
        }
        let active_refs = caps.max_active_reference_pictures.min(16);

        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: creating session ({}x{})", max_w, max_h);
        }
        let mut video_session = VideoSession::new_h264_decode(
            // SAFETY: extending `&device` lifetime to 'static. The
            // `Device` is owned by `Self` and freed only after the
            // session in Drop ordering — same struct.
            unsafe { &*(&device as *const Device) },
            pd,
            qfi,
            &caps,
            (max_w, max_h),
            profile_idc,
            dpb_slots,
            active_refs,
        )
        .map_err(|e| Error::unsupported(format!("vulkan-video: vkCreateVideoSessionKHR: {e}")))?;
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: session created, binding memory");
        }
        video_session
            .allocate_and_bind_memory(pd)
            .map_err(|e| Error::other(format!("vulkan-video: bind session memory: {e}")))?;
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: session memory bound");
        }

        drop(pds);

        // ── VkVideoSessionParametersKHR (SPS + PPS) ─────────────
        let std_sps = std_sps_from_parsed(sps);
        let std_pps = std_pps_from_parsed(pps);

        let h264_params_add = VkVideoDecodeH264SessionParametersAddInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
            p_next: ptr::null(),
            std_sps_count: 1,
            p_std_sp_ss: &std_sps,
            std_pps_count: 1,
            p_std_pp_ss: &std_pps,
        };
        let h264_params_create = VkVideoDecodeH264SessionParametersCreateInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
            p_next: ptr::null(),
            max_std_sps_count: 1,
            max_std_pps_count: 1,
            p_parameters_add_info: &h264_params_add,
        };
        let params_create = VkVideoSessionParametersCreateInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
            p_next: &h264_params_create as *const _ as *const c_void,
            flags: 0,
            video_session_parameters_template: ptr::null_mut(),
            video_session: video_session.handle(),
        };
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: creating session parameters");
        }
        let mut session_params: VkVideoSessionParametersKHR = ptr::null_mut();
        let r = unsafe {
            (device.fns().create_video_session_parameters_khr)(
                device.handle(),
                &params_create,
                ptr::null(),
                &mut session_params,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkCreateVideoSessionParametersKHR", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: session parameters OK");
        }

        // ── Profile struct kept for image creation pNext chain ──
        let h264_profile = VkVideoDecodeH264ProfileInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR,
            p_next: ptr::null(),
            std_profile_idc: profile_idc,
            picture_layout: VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR,
        };
        let profile = VkVideoProfileInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
            p_next: &h264_profile as *const _ as *const c_void,
            video_codec_operation: VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
            chroma_subsampling: VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
            luma_bit_depth: VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
            chroma_bit_depth: VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
        };
        let profile_list = VkVideoProfileListInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR,
            p_next: ptr::null(),
            profile_count: 1,
            p_profiles: &profile,
        };

        // ── DPB image (2D array, layer per DPB slot) ────────────
        // When DPB and output coincide we also use this image as the
        // source of the post-decode `vkCmdCopyImageToBuffer`, so
        // `VK_IMAGE_USAGE_TRANSFER_SRC_BIT` has to be on the usage
        // flags (VUID-vkCmdCopyImageToBuffer-srcImage-00186 +
        // VUID-VkImageMemoryBarrier-oldLayout-01212).
        let dpb_layers = dpb_slots;
        let dpb_usage = if coincide {
            VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR
                | VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR
                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        } else {
            VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR
        };
        let dpb_image_ci = VkImageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            p_next: &profile_list as *const _ as *const c_void,
            flags: 0,
            image_type: VK_IMAGE_TYPE_2D,
            format: VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
            extent: VkExtent3D {
                width: max_w,
                height: max_h,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: dpb_layers,
            samples: VK_SAMPLE_COUNT_1_BIT,
            tiling: VK_IMAGE_TILING_OPTIMAL,
            usage: dpb_usage,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            initial_layout: VK_IMAGE_LAYOUT_UNDEFINED,
        };
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: creating DPB image");
        }
        let mut dpb_image: VkImage = ptr::null_mut();
        let r = unsafe {
            (device.fns().create_image)(device.handle(), &dpb_image_ci, ptr::null(), &mut dpb_image)
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkCreateImage(DPB)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: DPB image OK");
        }

        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: DPB image mem requirements");
        }
        let dpb_mem_reqs = {
            let mut req = VkMemoryRequirements::default();
            unsafe {
                (device.fns().get_image_memory_requirements)(device.handle(), dpb_image, &mut req)
            };
            req
        };
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: DPB mem reqs size={} type_bits=0x{:x}",
                dpb_mem_reqs.size, dpb_mem_reqs.memory_type_bits
            );
        }
        let dpb_type = pick_memory_type(
            &mem_props,
            dpb_mem_reqs.memory_type_bits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        .ok_or_else(|| Error::other("vulkan-video: no device-local memory type for DPB"))?;
        let dpb_alloc_info = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: dpb_mem_reqs.size,
            memory_type_index: dpb_type,
        };
        let mut dpb_memory: VkDeviceMemory = ptr::null_mut();
        let r = unsafe {
            (device.fns().allocate_memory)(
                device.handle(),
                &dpb_alloc_info,
                ptr::null(),
                &mut dpb_memory,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkAllocateMemory(DPB)", r));
        }
        let r =
            unsafe { (device.fns().bind_image_memory)(device.handle(), dpb_image, dpb_memory, 0) };
        if r != VK_SUCCESS {
            return Err(vk_err("vkBindImageMemory(DPB)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: DPB image+memory bound");
        }

        // DPB image view — base layer 0 (our setup slot is always 0).
        // For a multi-planar format (NV12 = G8_B8R8_2PLANE_420_UNORM)
        // an image view that covers all planes uses
        // `VK_IMAGE_ASPECT_COLOR_BIT`. The per-plane bits
        // (PLANE_0_BIT / PLANE_1_BIT) are reserved for plane-disjoint
        // views; specifying both is invalid
        // (VUID-VkImageViewCreateInfo-subresourceRange-07818 — only
        // one plane bit is permitted per view).
        let dpb_view_ci = VkImageViewCreateInfo {
            s_type: VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            image: dpb_image,
            view_type: VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            format: VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
            components: VkComponentMapping::default(),
            subresource_range: VkImageSubresourceRange {
                aspect_mask: VK_IMAGE_ASPECT_COLOR_BIT,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: dpb_layers,
            },
        };
        let mut dpb_image_view: VkImageView = ptr::null_mut();
        let r = unsafe {
            (device.fns().create_image_view)(
                device.handle(),
                &dpb_view_ci,
                ptr::null(),
                &mut dpb_image_view,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkCreateImageView(DPB)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: DPB image view OK");
        }

        // ── Output image: same as DPB on coincide drivers ───────
        let (output_image, output_memory, output_image_view) = if coincide {
            // Reuse — but we need a separate single-layer view for the
            // dst_picture_resource (NVIDIA accepts the array view though).
            // Same multi-planar aspect-mask rule as the DPB view above:
            // VK_IMAGE_ASPECT_COLOR_BIT for the all-planes view.
            let output_view_ci = VkImageViewCreateInfo {
                s_type: VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                image: dpb_image,
                view_type: VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                format: VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                components: VkComponentMapping::default(),
                subresource_range: VkImageSubresourceRange {
                    aspect_mask: VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: dpb_layers,
                },
            };
            let mut v: VkImageView = ptr::null_mut();
            let r = unsafe {
                (device.fns().create_image_view)(
                    device.handle(),
                    &output_view_ci,
                    ptr::null(),
                    &mut v,
                )
            };
            if r != VK_SUCCESS {
                return Err(vk_err("vkCreateImageView(output coincide)", r));
            }
            (dpb_image, dpb_memory, v)
        } else {
            let out_image_ci = VkImageCreateInfo {
                s_type: VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                p_next: &profile_list as *const _ as *const c_void,
                flags: 0,
                image_type: VK_IMAGE_TYPE_2D,
                format: VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                extent: VkExtent3D {
                    width: max_w,
                    height: max_h,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: VK_SAMPLE_COUNT_1_BIT,
                tiling: VK_IMAGE_TILING_OPTIMAL,
                usage: VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
                queue_family_index_count: 0,
                p_queue_family_indices: ptr::null(),
                initial_layout: VK_IMAGE_LAYOUT_UNDEFINED,
            };
            let mut img: VkImage = ptr::null_mut();
            let r = unsafe {
                (device.fns().create_image)(device.handle(), &out_image_ci, ptr::null(), &mut img)
            };
            if r != VK_SUCCESS {
                return Err(vk_err("vkCreateImage(output)", r));
            }
            let mut req = VkMemoryRequirements::default();
            unsafe {
                (device.fns().get_image_memory_requirements)(device.handle(), img, &mut req);
            }
            let t = pick_memory_type(
                &mem_props,
                req.memory_type_bits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            )
            .ok_or_else(|| Error::other("vulkan-video: no device-local for output image"))?;
            let alloc = VkMemoryAllocateInfo {
                s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                p_next: ptr::null(),
                allocation_size: req.size,
                memory_type_index: t,
            };
            let mut m: VkDeviceMemory = ptr::null_mut();
            let r = unsafe {
                (device.fns().allocate_memory)(device.handle(), &alloc, ptr::null(), &mut m)
            };
            if r != VK_SUCCESS {
                return Err(vk_err("vkAllocateMemory(output)", r));
            }
            let r = unsafe { (device.fns().bind_image_memory)(device.handle(), img, m, 0) };
            if r != VK_SUCCESS {
                return Err(vk_err("vkBindImageMemory(output)", r));
            }
            let view_ci = VkImageViewCreateInfo {
                s_type: VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                image: img,
                view_type: VK_IMAGE_VIEW_TYPE_2D,
                format: VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                components: VkComponentMapping::default(),
                subresource_range: VkImageSubresourceRange {
                    aspect_mask: VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            let mut v: VkImageView = ptr::null_mut();
            let r = unsafe {
                (device.fns().create_image_view)(device.handle(), &view_ci, ptr::null(), &mut v)
            };
            if r != VK_SUCCESS {
                return Err(vk_err("vkCreateImageView(output)", r));
            }
            (img, m, v)
        };

        // ── Bitstream buffer (host-visible) ─────────────────────
        //
        // Real 1080p IDR access units routinely exceed 64 KiB. Size this from
        // the coded picture rather than a fixture-era constant; two bytes per
        // coded pixel plus a 4 MiB floor leaves ample headroom for compressed
        // picture overhead while keeping even 4K allocations modest.
        let bitstream_size =
            bitstream_buffer_capacity(max_w, max_h, caps.min_bitstream_buffer_size_alignment);
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: creating bitstream buffer size={} (coincide={})",
                bitstream_size, coincide
            );
        }
        let buffer_ci = VkBufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: &profile_list as *const _ as *const c_void,
            flags: 0,
            size: bitstream_size,
            usage: VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };
        let mut bitstream_buffer: VkBuffer = ptr::null_mut();
        let r = unsafe {
            (device.fns().create_buffer)(
                device.handle(),
                &buffer_ci,
                ptr::null(),
                &mut bitstream_buffer,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkCreateBuffer(bitstream)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: bitstream buffer OK");
        }
        let mut bitstream_req = VkMemoryRequirements::default();
        unsafe {
            (device.fns().get_buffer_memory_requirements)(
                device.handle(),
                bitstream_buffer,
                &mut bitstream_req,
            );
        }
        let bitstream_t = pick_memory_type(
            &mem_props,
            bitstream_req.memory_type_bits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        .ok_or_else(|| Error::other("vulkan-video: no host-coherent memory for bitstream"))?;
        let alloc = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: bitstream_req.size,
            memory_type_index: bitstream_t,
        };
        let mut bitstream_memory: VkDeviceMemory = ptr::null_mut();
        let r = unsafe {
            (device.fns().allocate_memory)(
                device.handle(),
                &alloc,
                ptr::null(),
                &mut bitstream_memory,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkAllocateMemory(bitstream)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: bitstream memory allocated");
        }
        let r = unsafe {
            (device.fns().bind_buffer_memory)(
                device.handle(),
                bitstream_buffer,
                bitstream_memory,
                0,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkBindBufferMemory(bitstream)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: bitstream memory bound");
        }

        // ── Staging buffer (host-visible, big enough for NV12) ──
        //
        // NV12 = `VK_FORMAT_G8_B8R8_2PLANE_420_UNORM`:
        //   * plane 0 (luma) — 1 byte per texel, width × height texels
        //   * plane 1 (chroma) — 2 bytes per texel (R8G8 interleaved
        //     U+V), (width/2) × (height/2) texels
        //
        // `buffer_row_length` in `VkBufferImageCopy` is measured in
        // *texels* of the source plane, not bytes. So plane-1's
        // tightly-packed row length is width/2 texels (= width bytes
        // since each chroma texel is 2 bytes). Total staging size
        // therefore is `width*height + (width/2)*(height/2)*2` =
        // `width*height + width*height/2` = `width*height*3/2` —
        // standard 4:2:0 byte budget.
        //
        // Previously we used `chroma_stride = max_w` (i.e. row length
        // = full width texels, 2 bytes/texel) which made the GPU
        // address-space stride for plane 1 twice the real packed
        // stride and overran the staging buffer by ~38400 bytes
        // (VUID-vkCmdCopyImageToBuffer-pRegions-00183).
        let luma_stride = max_w;
        let chroma_stride_texels = max_w.div_ceil(2);
        let staging_chroma_height = max_h.div_ceil(2);
        // bytes-per-row for staging budget: luma 1 byte/texel,
        // chroma 2 bytes/texel. Allocate for the full coded extent; the
        // readback region itself may be cropped to smaller display dimensions.
        let staging_size = (luma_stride as u64) * (max_h as u64)
            + (chroma_stride_texels as u64) * 2 * (staging_chroma_height as u64);
        let staging_size = staging_size.max(1024);

        let staging_ci = VkBufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            size: staging_size,
            usage: VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: creating staging buffer (size={})",
                staging_size
            );
        }
        let mut staging_buffer: VkBuffer = ptr::null_mut();
        let r = unsafe {
            (device.fns().create_buffer)(
                device.handle(),
                &staging_ci,
                ptr::null(),
                &mut staging_buffer,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkCreateBuffer(staging)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: staging buffer OK");
        }
        let mut staging_req = VkMemoryRequirements::default();
        unsafe {
            (device.fns().get_buffer_memory_requirements)(
                device.handle(),
                staging_buffer,
                &mut staging_req,
            );
        }
        let staging_t = pick_memory_type(
            &mem_props,
            staging_req.memory_type_bits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        .ok_or_else(|| Error::other("vulkan-video: no host-coherent memory for staging"))?;
        let alloc = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: staging_req.size,
            memory_type_index: staging_t,
        };
        let mut staging_memory: VkDeviceMemory = ptr::null_mut();
        let r = unsafe {
            (device.fns().allocate_memory)(
                device.handle(),
                &alloc,
                ptr::null(),
                &mut staging_memory,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkAllocateMemory(staging)", r));
        }
        let r = unsafe {
            (device.fns().bind_buffer_memory)(device.handle(), staging_buffer, staging_memory, 0)
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkBindBufferMemory(staging)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: staging buffer bound");
        }

        // ── Command pool + command buffer ──────────────────────
        let cp_ci = VkCommandPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queue_family_index: qfi,
        };
        let mut command_pool: VkCommandPool = ptr::null_mut();
        let r = unsafe {
            (device.fns().create_command_pool)(
                device.handle(),
                &cp_ci,
                ptr::null(),
                &mut command_pool,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkCreateCommandPool", r));
        }
        let cb_ai = VkCommandBufferAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut command_buffer: VkCommandBuffer = ptr::null_mut();
        let r = unsafe {
            (device.fns().allocate_command_buffers)(device.handle(), &cb_ai, &mut command_buffer)
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkAllocateCommandBuffers", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: command pool + buffer OK");
        }
        let (display_width, display_height) = display_dimensions(sps);
        let (output_width, output_height, output_chroma_height) =
            output_dimensions(display_width, display_height, max_w, max_h);
        Ok(Self {
            command_buffer,
            command_pool,
            bitstream_buffer,
            bitstream_memory,
            bitstream_size,
            staging_buffer,
            staging_memory,
            staging_size,
            output_image_view,
            output_image,
            output_memory,
            dpb_image_view,
            dpb_image,
            dpb_memory,
            session_params,
            // SAFETY: extends the `'device` borrow on `video_session` to
            // `'static`; the session is owned by `Self` and dropped before
            // the `Device` it borrows, so the apparent lifetime is sound.
            session: Some(unsafe {
                std::mem::transmute::<VideoSession<'_>, VideoSession<'static>>(video_session)
            }),
            queue_family_index: qfi,
            queue_index,
            physical_device_handle: pd_handle,
            device,
            instance,
            width: output_width,
            height: output_height,
            luma_stride,
            chroma_stride: chroma_stride_texels,
            chroma_height: output_chroma_height,
            coincide,
            dpb_slot_count: dpb_slots,
            dpb_initialized: false,
            output_initialized: false,
            needs_reset: true,
            bitstream_offset_alignment: caps.min_bitstream_buffer_offset_alignment.max(1),
            bitstream_size_alignment: caps.min_bitstream_buffer_size_alignment.max(1),
        })
    }

    fn decode_picture(
        &mut self,
        bitstream: &[u8],
        picture: &PreparedH264Picture,
        target_slot: u32,
        target_was_active: bool,
        slot_map: &HashMap<u32, u32>,
        reset_session: bool,
    ) -> Result<VideoFrame> {
        if target_slot >= self.dpb_slot_count {
            return Err(Error::invalid(format!(
                "vulkan-video: target DPB slot {target_slot} exceeds {} allocated slots",
                self.dpb_slot_count
            )));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: decode_picture bytes={} frame_num={} poc={} refs={} target_slot={} idr={}",
                bitstream.len(),
                picture.header.frame_num,
                picture.poc.pic_order_cnt,
                picture.references.len(),
                target_slot,
                picture.is_idr()
            );
        }

        if (bitstream.len() as u64) > self.bitstream_size {
            return Err(Error::other(format!(
                "vulkan-video: bitstream {} > buffer {}",
                bitstream.len(),
                self.bitstream_size
            )));
        }
        unsafe {
            let mut p: *mut c_void = ptr::null_mut();
            let r = (self.device.fns().map_memory)(
                self.device.handle(),
                self.bitstream_memory,
                0,
                self.bitstream_size,
                0,
                &mut p,
            );
            if r != VK_SUCCESS {
                return Err(vk_err("vkMapMemory(bitstream)", r));
            }
            std::ptr::write_bytes(p as *mut u8, 0, self.bitstream_size as usize);
            std::ptr::copy_nonoverlapping(bitstream.as_ptr(), p as *mut u8, bitstream.len());
            (self.device.fns().unmap_memory)(self.device.handle(), self.bitstream_memory);
        }

        let (coded_width, coded_height) = coded_dimensions(&picture.sps);
        let std_pic_info = std_picture_info(picture);
        let slice_offsets = compute_slice_offsets(bitstream);
        if slice_offsets.is_empty() {
            return Err(Error::invalid(
                "vulkan-video: H.264 access unit contains no VCL slice offsets",
            ));
        }
        if slice_offsets.len() as u32 != picture.slice_count {
            return Err(Error::invalid(format!(
                "vulkan-video: parser reported {} slices but Vulkan bitstream scan found {}",
                picture.slice_count,
                slice_offsets.len()
            )));
        }
        let h264_pic_info = VkVideoDecodeH264PictureInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR,
            p_next: ptr::null(),
            p_std_picture_info: &std_pic_info,
            slice_count: slice_offsets.len() as u32,
            p_slice_offsets: slice_offsets.as_ptr(),
        };

        let setup_resource = VkVideoPictureResourceInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
            p_next: ptr::null(),
            coded_offset: VkOffset2D { x: 0, y: 0 },
            coded_extent: VkExtent2D {
                width: coded_width,
                height: coded_height,
            },
            base_array_layer: target_slot,
            image_view_binding: self.dpb_image_view,
        };
        let setup_std_ref = std_current_reference_info(picture);
        let setup_h264_slot = VkVideoDecodeH264DpbSlotInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
            p_next: ptr::null(),
            p_std_reference_info: &setup_std_ref,
        };
        let setup_ref = VkVideoReferenceSlotInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
            p_next: &setup_h264_slot as *const _ as *const c_void,
            slot_index: target_slot as i32,
            p_picture_resource: &setup_resource,
        };

        let mut mapped_refs: Vec<(&DpbEntry, u32)> = Vec::new();
        for reference in &picture.references {
            if reference.marking == RefMarking::Unused {
                continue;
            }
            let slot = slot_map.get(&reference.dpb_key).copied().ok_or_else(|| {
                Error::other(format!(
                    "vulkan-video: live H.264 DPB key {} has no Vulkan slot",
                    reference.dpb_key
                ))
            })?;
            mapped_refs.push((reference, slot));
        }

        let std_refs: Vec<StdVideoDecodeH264ReferenceInfo> = mapped_refs
            .iter()
            .map(|(reference, _)| std_reference_info(reference))
            .collect();
        let ref_resources: Vec<VkVideoPictureResourceInfoKHR> = mapped_refs
            .iter()
            .map(|(_, slot)| VkVideoPictureResourceInfoKHR {
                s_type: VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: ptr::null(),
                coded_offset: VkOffset2D { x: 0, y: 0 },
                coded_extent: VkExtent2D {
                    width: coded_width,
                    height: coded_height,
                },
                base_array_layer: *slot,
                image_view_binding: self.dpb_image_view,
            })
            .collect();
        let ref_h264_slots: Vec<VkVideoDecodeH264DpbSlotInfoKHR> = std_refs
            .iter()
            .map(|std_ref| VkVideoDecodeH264DpbSlotInfoKHR {
                s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
                p_next: ptr::null(),
                p_std_reference_info: std_ref,
            })
            .collect();
        let ref_slots: Vec<VkVideoReferenceSlotInfoKHR> = mapped_refs
            .iter()
            .enumerate()
            .map(|(index, (_, slot))| VkVideoReferenceSlotInfoKHR {
                s_type: VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: &ref_h264_slots[index] as *const _ as *const c_void,
                slot_index: *slot as i32,
                p_picture_resource: &ref_resources[index],
            })
            .collect();
        let reference_slot_ptr = if ref_slots.is_empty() {
            ptr::null()
        } else {
            ref_slots.as_ptr()
        };

        // vkCmdBeginVideoCodingKHR binds every resource that may be used in
        // this coding scope. Existing references keep their current slot
        // associations. A reconstruction target that is not active yet is
        // bound with slotIndex=-1 so vkCmdDecodeVideoKHR can subsequently
        // activate it through pSetupReferenceSlot. If the physical DPB layer
        // is being recycled while Vulkan still considers its old slot active,
        // bind that existing association first; the decode operation replaces
        // it with the reconstructed picture.
        let mut begin_resources: Vec<VkVideoPictureResourceInfoKHR> = mapped_refs
            .iter()
            .map(|(_, slot)| VkVideoPictureResourceInfoKHR {
                s_type: VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: ptr::null(),
                coded_offset: VkOffset2D { x: 0, y: 0 },
                coded_extent: VkExtent2D {
                    width: coded_width,
                    height: coded_height,
                },
                base_array_layer: *slot,
                image_view_binding: self.dpb_image_view,
            })
            .collect();
        begin_resources.push(setup_resource);

        let mut begin_slots: Vec<VkVideoReferenceSlotInfoKHR> = mapped_refs
            .iter()
            .enumerate()
            .map(|(index, (_, slot))| VkVideoReferenceSlotInfoKHR {
                s_type: VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: ptr::null(),
                slot_index: *slot as i32,
                p_picture_resource: &begin_resources[index],
            })
            .collect();
        let target_resource_index = begin_resources.len() - 1;
        begin_slots.push(VkVideoReferenceSlotInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
            p_next: ptr::null(),
            slot_index: if target_was_active {
                target_slot as i32
            } else {
                -1
            },
            p_picture_resource: &begin_resources[target_resource_index],
        });

        let output_layer = if self.coincide { target_slot } else { 0 };
        let dst_picture_resource = VkVideoPictureResourceInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
            p_next: ptr::null(),
            coded_offset: VkOffset2D { x: 0, y: 0 },
            coded_extent: VkExtent2D {
                width: coded_width,
                height: coded_height,
            },
            base_array_layer: output_layer,
            image_view_binding: self.output_image_view,
        };

        let session_handle = self
            .session
            .as_ref()
            .ok_or_else(|| Error::other("vulkan-video: no session"))?
            .handle();
        let begin_info = VkVideoBeginCodingInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
            p_next: ptr::null(),
            flags: 0,
            video_session: session_handle,
            video_session_parameters: self.session_params,
            reference_slot_count: begin_slots.len() as u32,
            p_reference_slots: begin_slots.as_ptr(),
        };
        let control_info = VkVideoCodingControlInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
            p_next: ptr::null(),
            flags: VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR,
        };
        let decode_info = VkVideoDecodeInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR,
            p_next: &h264_pic_info as *const _ as *const c_void,
            flags: 0,
            src_buffer: self.bitstream_buffer,
            src_buffer_offset: 0,
            src_buffer_range: align_up(bitstream.len() as u64, self.bitstream_size_alignment),
            dst_picture_resource,
            p_setup_reference_slot: &setup_ref,
            reference_slot_count: ref_slots.len() as u32,
            p_reference_slots: reference_slot_ptr,
        };
        let end_info = VkVideoEndCodingInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR,
            p_next: ptr::null(),
            flags: 0,
        };

        let cb_begin = VkCommandBufferBeginInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: ptr::null(),
        };
        let r = unsafe { (self.device.fns().begin_command_buffer)(self.command_buffer, &cb_begin) };
        if r != VK_SUCCESS {
            return Err(vk_err("vkBeginCommandBuffer", r));
        }

        let mut initial_barriers = Vec::new();
        if !self.dpb_initialized {
            initial_barriers.push(VkImageMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: 0,
                dst_access_mask: VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                old_layout: VK_IMAGE_LAYOUT_UNDEFINED,
                new_layout: VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
                src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                image: self.dpb_image,
                subresource_range: VkImageSubresourceRange {
                    aspect_mask: VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: self.dpb_slot_count,
                },
            });
        }
        if !self.coincide && !self.output_initialized {
            initial_barriers.push(VkImageMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: 0,
                dst_access_mask: VK_ACCESS_MEMORY_WRITE_BIT,
                old_layout: VK_IMAGE_LAYOUT_UNDEFINED,
                new_layout: VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR,
                src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                image: self.output_image,
                subresource_range: VkImageSubresourceRange {
                    aspect_mask: VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            });
        }
        if !initial_barriers.is_empty() {
            unsafe {
                (self.device.fns().cmd_pipeline_barrier)(
                    self.command_buffer,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    0,
                    0,
                    ptr::null(),
                    0,
                    ptr::null(),
                    initial_barriers.len() as u32,
                    initial_barriers.as_ptr(),
                );
            }
        }

        let skip_decode = std::env::var("OXIDEAV_VK_SKIP_DECODE").is_ok();
        unsafe {
            (self.device.fns().cmd_begin_video_coding_khr)(self.command_buffer, &begin_info);
            if reset_session {
                (self.device.fns().cmd_control_video_coding_khr)(
                    self.command_buffer,
                    &control_info,
                );
            }
            if !skip_decode {
                (self.device.fns().cmd_decode_video_khr)(self.command_buffer, &decode_info);
            }
            (self.device.fns().cmd_end_video_coding_khr)(self.command_buffer, &end_info);
        }

        let source_layout = if self.coincide {
            VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR
        } else {
            VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR
        };
        let to_transfer = VkImageMemoryBarrier {
            s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: VK_ACCESS_MEMORY_WRITE_BIT,
            dst_access_mask: VK_ACCESS_TRANSFER_READ_BIT,
            old_layout: source_layout,
            new_layout: VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
            image: self.output_image,
            subresource_range: VkImageSubresourceRange {
                aspect_mask: VK_IMAGE_ASPECT_COLOR_BIT,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: output_layer,
                layer_count: 1,
            },
        };
        unsafe {
            (self.device.fns().cmd_pipeline_barrier)(
                self.command_buffer,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                ptr::null(),
                0,
                ptr::null(),
                1,
                &to_transfer,
            );
        }

        let luma_bytes = (self.luma_stride as u64) * (self.height as u64);
        let regions = [
            VkBufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: self.luma_stride,
                buffer_image_height: self.height,
                image_subresource: VkImageSubresourceLayers {
                    aspect_mask: VK_IMAGE_ASPECT_PLANE_0_BIT,
                    mip_level: 0,
                    base_array_layer: output_layer,
                    layer_count: 1,
                },
                image_offset: VkOffset3D::default(),
                image_extent: VkExtent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                },
            },
            VkBufferImageCopy {
                buffer_offset: luma_bytes,
                buffer_row_length: self.chroma_stride,
                buffer_image_height: self.chroma_height,
                image_subresource: VkImageSubresourceLayers {
                    aspect_mask: VK_IMAGE_ASPECT_PLANE_1_BIT,
                    mip_level: 0,
                    base_array_layer: output_layer,
                    layer_count: 1,
                },
                image_offset: VkOffset3D::default(),
                image_extent: VkExtent3D {
                    width: self.width.div_ceil(2),
                    height: self.chroma_height,
                    depth: 1,
                },
            },
        ];
        unsafe {
            (self.device.fns().cmd_copy_image_to_buffer)(
                self.command_buffer,
                self.output_image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                self.staging_buffer,
                regions.len() as u32,
                regions.as_ptr(),
            );
        }

        let quiescent_layout = if self.coincide {
            VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR
        } else {
            VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR
        };
        let to_quiescent = VkImageMemoryBarrier {
            s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: VK_ACCESS_TRANSFER_READ_BIT,
            dst_access_mask: VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            old_layout: VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            new_layout: quiescent_layout,
            src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
            image: self.output_image,
            subresource_range: VkImageSubresourceRange {
                aspect_mask: VK_IMAGE_ASPECT_COLOR_BIT,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: output_layer,
                layer_count: 1,
            },
        };
        unsafe {
            (self.device.fns().cmd_pipeline_barrier)(
                self.command_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0,
                0,
                ptr::null(),
                0,
                ptr::null(),
                1,
                &to_quiescent,
            );
        }

        let r = unsafe { (self.device.fns().end_command_buffer)(self.command_buffer) };
        if r != VK_SUCCESS {
            return Err(vk_err("vkEndCommandBuffer", r));
        }
        if std::env::var("OXIDEAV_VK_SKIP_SUBMIT").is_ok() {
            return Err(Error::other("OXIDEAV_VK_SKIP_SUBMIT set; skipping submit"));
        }

        let queue = self
            .device
            .queue_indexed(self.queue_family_index, self.queue_index);
        let submit = VkSubmitInfo {
            s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffer,
            signal_semaphore_count: 0,
            p_signal_semaphores: ptr::null(),
        };
        let r = unsafe {
            (self.device.fns().queue_submit)(queue.handle(), 1, &submit, ptr::null_mut())
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkQueueSubmit", r));
        }
        let r = unsafe { (self.device.fns().queue_wait_idle)(queue.handle()) };
        if r != VK_SUCCESS {
            return Err(vk_err("vkQueueWaitIdle", r));
        }
        self.dpb_initialized = true;
        if !self.coincide {
            self.output_initialized = true;
        }
        self.needs_reset = false;

        let width = self.width as usize;
        let height = self.height as usize;
        let lstride = self.luma_stride as usize;
        let cw = self.width.div_ceil(2) as usize;
        let ch = self.chroma_height as usize;
        let cstride_bytes = (self.chroma_stride as usize) * 2;
        let chroma_off = lstride * height;

        let mut frame_y = vec![0u8; width * height];
        let mut cached_uv = vec![0u8; cstride_bytes * ch];

        unsafe {
            let mut p: *mut c_void = ptr::null_mut();
            let r = (self.device.fns().map_memory)(
                self.device.handle(),
                self.staging_memory,
                0,
                self.staging_size,
                0,
                &mut p,
            );
            if r != VK_SUCCESS {
                return Err(vk_err("vkMapMemory(staging)", r));
            }
            let host = p as *const u8;

            // Host-visible GPU memory can be effectively uncached for fine-grained
            // CPU reads. Bulk-copy each plane into ordinary cached RAM first;
            // deinterleaving UV directly from the mapped allocation was the
            // dominant 720p60 cost on NVIDIA (tens of milliseconds per frame).
            if lstride == width {
                std::ptr::copy_nonoverlapping(host, frame_y.as_mut_ptr(), frame_y.len());
            } else {
                for y in 0..height {
                    std::ptr::copy_nonoverlapping(
                        host.add(y * lstride),
                        frame_y.as_mut_ptr().add(y * width),
                        width,
                    );
                }
            }
            std::ptr::copy_nonoverlapping(
                host.add(chroma_off),
                cached_uv.as_mut_ptr(),
                cached_uv.len(),
            );
            (self.device.fns().unmap_memory)(self.device.handle(), self.staging_memory);
        }

        let mut frame_u = vec![0u8; cw * ch];
        let mut frame_v = vec![0u8; cw * ch];
        for y in 0..ch {
            let src_row = &cached_uv[y * cstride_bytes..y * cstride_bytes + cw * 2];
            let u_row = &mut frame_u[y * cw..(y + 1) * cw];
            let v_row = &mut frame_v[y * cw..(y + 1) * cw];
            for (x, pair) in src_row.chunks_exact(2).enumerate() {
                u_row[x] = pair[0];
                v_row[x] = pair[1];
            }
        }

        Ok(VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: self.width as usize,
                    data: frame_y,
                },
                VideoPlane {
                    stride: cw,
                    data: frame_u,
                },
                VideoPlane {
                    stride: cw,
                    data: frame_v,
                },
            ],
        })
    }
}

fn validate_picture_shape(picture: &PreparedH264Picture) -> Result<()> {
    let sps = &picture.sps;
    let pps = &picture.pps;
    let header = &picture.header;
    if header.field_pic_flag || sps.mb_adaptive_frame_field_flag {
        return Err(Error::unsupported(
            "vulkan-video: H.264 streaming decoder currently supports progressive frame pictures only",
        ));
    }
    if sps.chroma_format_idc != 1
        || sps.separate_colour_plane_flag
        || sps.bit_depth_luma_minus8 != 0
        || sps.bit_depth_chroma_minus8 != 0
    {
        return Err(Error::unsupported(
            "vulkan-video: H.264 streaming decoder currently supports 8-bit 4:2:0 only",
        ));
    }
    if !matches!(sps.profile_idc, 66 | 77 | 100) {
        return Err(Error::unsupported(format!(
            "vulkan-video: unsupported H.264 profile_idc={}",
            sps.profile_idc
        )));
    }
    if sps.seq_scaling_matrix_present_flag
        || pps
            .extension
            .as_ref()
            .is_some_and(|e| e.pic_scaling_matrix_present_flag)
    {
        return Err(Error::unsupported(
            "vulkan-video: custom H.264 scaling matrices are not translated yet",
        ));
    }
    if pps.num_slice_groups_minus1 != 0 {
        return Err(Error::unsupported(
            "vulkan-video: H.264 flexible macroblock ordering is not supported yet",
        ));
    }
    if sps
        .frame_cropping
        .as_ref()
        .is_some_and(|crop| crop.left != 0 || crop.top != 0)
    {
        return Err(Error::unsupported(
            "vulkan-video: non-zero left/top H.264 frame cropping is not supported yet",
        ));
    }
    Ok(())
}

fn coded_dimensions(sps: &Sps) -> (u32, u32) {
    (sps.pic_width_in_mbs() * 16, sps.frame_height_in_mbs() * 16)
}

fn display_dimensions(sps: &Sps) -> (u32, u32) {
    let (coded_width, coded_height) = coded_dimensions(sps);
    let Some(crop) = &sps.frame_cropping else {
        return (coded_width, coded_height);
    };
    // This backend currently accepts only progressive 4:2:0, for which
    // CropUnitX=2 and CropUnitY=2.
    (
        coded_width.saturating_sub(2 * (crop.left + crop.right)),
        coded_height.saturating_sub(2 * (crop.top + crop.bottom)),
    )
}

fn h264_profile_idc(profile_idc: u8) -> Result<sys::StdVideoH264ProfileIdc> {
    match profile_idc {
        66 => Ok(sys::STD_VIDEO_H264_PROFILE_IDC_BASELINE),
        77 => Ok(sys::STD_VIDEO_H264_PROFILE_IDC_MAIN),
        100 => Ok(sys::STD_VIDEO_H264_PROFILE_IDC_HIGH),
        other => Err(Error::unsupported(format!(
            "vulkan-video: unsupported H.264 profile_idc={other}"
        ))),
    }
}

fn std_picture_info(picture: &PreparedH264Picture) -> StdVideoDecodeH264PictureInfo {
    let mut flags = 0;
    if picture.header.field_pic_flag {
        flags |= StdVideoDecodeH264PictureInfoFlags::FIELD_PIC;
    }
    if picture.header.slice_type.is_intra() {
        flags |= StdVideoDecodeH264PictureInfoFlags::IS_INTRA;
    }
    if picture.is_idr() {
        flags |= StdVideoDecodeH264PictureInfoFlags::IDR_PIC;
    }
    if picture.header.bottom_field_flag {
        flags |= StdVideoDecodeH264PictureInfoFlags::BOTTOM_FIELD;
    }
    if picture.is_reference() {
        flags |= StdVideoDecodeH264PictureInfoFlags::IS_REFERENCE;
    }
    StdVideoDecodeH264PictureInfo {
        flags: StdVideoDecodeH264PictureInfoFlags { flags },
        seq_parameter_set_id: picture.sps.seq_parameter_set_id as u8,
        pic_parameter_set_id: picture.pps.pic_parameter_set_id as u8,
        reserved1: 0,
        reserved2: 0,
        frame_num: picture.header.frame_num as u16,
        idr_pic_id: picture.header.idr_pic_id as u16,
        pic_order_cnt: [
            picture.poc.top_field_order_cnt,
            picture.poc.bottom_field_order_cnt,
        ],
    }
}

fn std_reference_info(entry: &DpbEntry) -> StdVideoDecodeH264ReferenceInfo {
    let long_term = entry.marking == RefMarking::LongTerm;
    let mut flags = 0;
    if long_term {
        flags |= sys::StdVideoDecodeH264ReferenceInfoFlags::USED_FOR_LONG_TERM_REFERENCE;
    }
    StdVideoDecodeH264ReferenceInfo {
        flags: sys::StdVideoDecodeH264ReferenceInfoFlags { flags },
        frame_num: if long_term {
            entry.long_term_frame_idx as u16
        } else {
            entry.frame_num as u16
        },
        reserved: 0,
        pic_order_cnt: [entry.top_field_order_cnt, entry.bottom_field_order_cnt],
    }
}

fn current_long_term_index(picture: &PreparedH264Picture) -> Option<u32> {
    let marking = picture.header.dec_ref_pic_marking.as_ref()?;
    if picture.is_idr() {
        return marking.long_term_reference_flag.then_some(0);
    }
    marking
        .adaptive_marking
        .as_ref()?
        .iter()
        .find_map(|op| match op {
            oxideav_h264::slice_header::MmcoOp::AssignCurrentLongTerm(index) => Some(*index),
            _ => None,
        })
}

fn std_current_reference_info(picture: &PreparedH264Picture) -> StdVideoDecodeH264ReferenceInfo {
    let long_term_index = current_long_term_index(picture);
    let mut flags = 0;
    if long_term_index.is_some() {
        flags |= sys::StdVideoDecodeH264ReferenceInfoFlags::USED_FOR_LONG_TERM_REFERENCE;
    }
    StdVideoDecodeH264ReferenceInfo {
        flags: sys::StdVideoDecodeH264ReferenceInfoFlags { flags },
        frame_num: long_term_index.unwrap_or(picture.header.frame_num) as u16,
        reserved: 0,
        pic_order_cnt: [
            picture.poc.top_field_order_cnt,
            picture.poc.bottom_field_order_cnt,
        ],
    }
}

// ─────────────────────── Sps/Pps → Std structs ──────────────────────

fn std_sps_from_parsed(s: &Sps) -> StdVideoH264SequenceParameterSet {
    let mut flags: u32 = 0;
    if s.constraint_set_flags & (1 << 0) != 0 {
        flags |= StdVideoH264SpsFlags::CONSTRAINT_SET0;
    }
    if s.constraint_set_flags & (1 << 1) != 0 {
        flags |= StdVideoH264SpsFlags::CONSTRAINT_SET1;
    }
    if s.constraint_set_flags & (1 << 2) != 0 {
        flags |= StdVideoH264SpsFlags::CONSTRAINT_SET2;
    }
    if s.constraint_set_flags & (1 << 3) != 0 {
        flags |= StdVideoH264SpsFlags::CONSTRAINT_SET3;
    }
    if s.constraint_set_flags & (1 << 4) != 0 {
        flags |= StdVideoH264SpsFlags::CONSTRAINT_SET4;
    }
    if s.constraint_set_flags & (1 << 5) != 0 {
        flags |= StdVideoH264SpsFlags::CONSTRAINT_SET5;
    }
    if s.direct_8x8_inference_flag {
        flags |= StdVideoH264SpsFlags::DIRECT_8X8_INFERENCE;
    }
    if s.mb_adaptive_frame_field_flag {
        flags |= StdVideoH264SpsFlags::MB_ADAPTIVE_FRAME_FIELD;
    }
    if s.frame_mbs_only_flag {
        flags |= StdVideoH264SpsFlags::FRAME_MBS_ONLY;
    }
    if s.delta_pic_order_always_zero_flag {
        flags |= StdVideoH264SpsFlags::DELTA_POC_ALWAYS_ZERO;
    }
    if s.separate_colour_plane_flag {
        flags |= StdVideoH264SpsFlags::SEPARATE_COLOUR_PLANE;
    }
    if s.gaps_in_frame_num_value_allowed_flag {
        flags |= StdVideoH264SpsFlags::GAPS_IN_FRAME_NUM;
    }
    if s.qpprime_y_zero_transform_bypass_flag {
        flags |= StdVideoH264SpsFlags::QPPRIME_Y_ZERO_TRANSFORM_BYPASS;
    }
    if s.frame_cropping.is_some() {
        flags |= StdVideoH264SpsFlags::FRAME_CROPPING;
    }

    let (crop_left, crop_right, crop_top, crop_bottom) = s
        .frame_cropping
        .as_ref()
        .map(|crop| (crop.left, crop.right, crop.top, crop.bottom))
        .unwrap_or((0, 0, 0, 0));
    let offset_ptr = if s.offset_for_ref_frame.is_empty() {
        ptr::null()
    } else {
        s.offset_for_ref_frame.as_ptr()
    };

    StdVideoH264SequenceParameterSet {
        flags: StdVideoH264SpsFlags { flags },
        profile_idc: s.profile_idc as i32,
        level_idc: h264_level_byte_to_idc(s.level_idc),
        chroma_format_idc: s.chroma_format_idc as i32,
        seq_parameter_set_id: s.seq_parameter_set_id as u8,
        bit_depth_luma_minus8: s.bit_depth_luma_minus8 as u8,
        bit_depth_chroma_minus8: s.bit_depth_chroma_minus8 as u8,
        log2_max_frame_num_minus4: s.log2_max_frame_num_minus4 as u8,
        pic_order_cnt_type: s.pic_order_cnt_type as i32,
        offset_for_non_ref_pic: s.offset_for_non_ref_pic,
        offset_for_top_to_bottom_field: s.offset_for_top_to_bottom_field,
        log2_max_pic_order_cnt_lsb_minus4: s.log2_max_pic_order_cnt_lsb_minus4 as u8,
        num_ref_frames_in_pic_order_cnt_cycle: s.num_ref_frames_in_pic_order_cnt_cycle as u8,
        max_num_ref_frames: s.max_num_ref_frames as u8,
        reserved1: 0,
        pic_width_in_mbs_minus1: s.pic_width_in_mbs_minus1,
        pic_height_in_map_units_minus1: s.pic_height_in_map_units_minus1,
        frame_crop_left_offset: crop_left,
        frame_crop_right_offset: crop_right,
        frame_crop_top_offset: crop_top,
        frame_crop_bottom_offset: crop_bottom,
        reserved2: 0,
        p_offset_for_ref_frame: offset_ptr,
        p_scaling_lists: ptr::null(),
        // VUI is not consumed by H.264 picture reconstruction; omit it until
        // the Vulkan std-video VUI structs are modelled by this crate.
        p_sequence_parameter_set_vui: ptr::null(),
    }
}

fn std_pps_from_parsed(p: &Pps) -> StdVideoH264PictureParameterSet {
    let mut flags: u32 = 0;
    if p.transform_8x8_mode_flag() {
        flags |= StdVideoH264PpsFlags::TRANSFORM_8X8_MODE;
    }
    if p.redundant_pic_cnt_present_flag {
        flags |= StdVideoH264PpsFlags::REDUNDANT_PIC_CNT;
    }
    if p.constrained_intra_pred_flag {
        flags |= StdVideoH264PpsFlags::CONSTRAINED_INTRA_PRED;
    }
    if p.deblocking_filter_control_present_flag {
        flags |= StdVideoH264PpsFlags::DEBLOCK_FILTER_CTRL;
    }
    if p.weighted_pred_flag {
        flags |= StdVideoH264PpsFlags::WEIGHTED_PRED;
    }
    if p.bottom_field_pic_order_in_frame_present_flag {
        flags |= StdVideoH264PpsFlags::BOTTOM_FIELD_POC_IN_FRAME;
    }
    if p.entropy_coding_mode_flag {
        flags |= StdVideoH264PpsFlags::ENTROPY_CODING_MODE;
    }
    StdVideoH264PictureParameterSet {
        flags: StdVideoH264PpsFlags { flags },
        seq_parameter_set_id: p.seq_parameter_set_id as u8,
        pic_parameter_set_id: p.pic_parameter_set_id as u8,
        num_ref_idx_l0_default_active_minus1: p.num_ref_idx_l0_default_active_minus1 as u8,
        num_ref_idx_l1_default_active_minus1: p.num_ref_idx_l1_default_active_minus1 as u8,
        weighted_bipred_idc: p.weighted_bipred_idc as i32,
        pic_init_qp_minus26: p.pic_init_qp_minus26 as i8,
        pic_init_qs_minus26: p.pic_init_qs_minus26 as i8,
        chroma_qp_index_offset: p.chroma_qp_index_offset as i8,
        second_chroma_qp_index_offset: p.second_chroma_qp_index_offset() as i8,
        p_scaling_lists: ptr::null(),
    }
}

/// Convert the H.264 raw `level_idc` byte (which encodes 30, 31, 41…
/// in the bitstream) to the contiguous `StdVideoH264LevelIdc` enum
/// value (1.0 → 0, 1.1 → 1, …, 5.1 → 14, …, 6.2 → 18).
fn h264_level_byte_to_idc(b: u8) -> sys::StdVideoH264LevelIdc {
    match b {
        10 => 0,  // 1.0
        11 => 1,  // 1.1
        12 => 2,  // 1.2
        13 => 3,  // 1.3
        20 => 4,  // 2.0
        21 => 5,  // 2.1
        22 => 6,  // 2.2
        30 => 7,  // 3.0
        31 => 8,  // 3.1
        32 => 9,  // 3.2
        40 => 10, // 4.0
        41 => 11, // 4.1
        42 => 12, // 4.2
        50 => 13, // 5.0
        51 => 14, // 5.1
        52 => 15, // 5.2
        60 => 16, // 6.0
        61 => 17, // 6.1
        62 => 18, // 6.2
        _ => 14,  // default to 5.1
    }
}

#[cfg(test)]
mod tests {
    use super::{bitstream_buffer_capacity, output_dimensions};

    #[test]
    fn bitstream_buffer_capacity_handles_real_1080p_access_units() {
        let capacity = bitstream_buffer_capacity(1920, 1088, 4096);
        assert!(capacity > 160_928);
        assert!(capacity >= 4 * 1024 * 1024);
        assert_eq!(capacity % 4096, 0);
    }

    #[test]
    fn cropped_output_geometry_uses_display_chroma_height() {
        assert_eq!(output_dimensions(1920, 1080, 1920, 1088), (1920, 1080, 540));
        assert_eq!(output_dimensions(284, 160, 288, 160), (284, 160, 80));
    }
}
