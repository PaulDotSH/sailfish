#![allow(clippy::cast_ptr_alignment)]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ptr;

use super::super::Buffer;
use super::super::utils::memcpy_16;
use super::naive::{commit, commit_and_reserve};
use super::{ESCAPED, ESCAPED_LEN, ESCAPED_PADDED, ESCAPE_LUT};

const VECTOR_BYTES: usize = std::mem::size_of::<__m128i>();

/// Room needed to escape a block of up to 32 bytes, besides the rest of the input:
/// each byte expands to at most 6 bytes, and the last entity may be written 2 bytes
/// past its end.
const BLOCK_RESERVE: usize = 32 * 6 + 2;

/// Flags the bytes of `x` that may need escaping (`#` included)
#[inline]
#[target_feature(enable = "sse2")]
fn maskgen(x: __m128i) -> u32 {
    let v_independent1 = _mm_set1_epi8(5);
    let v_independent2 = _mm_set1_epi8(2);
    let v_key1 = _mm_set1_epi8(0x27);
    let v_key2 = _mm_set1_epi8(0x3e);
    _mm_movemask_epi8(_mm_or_si128(
        _mm_cmpeq_epi8(_mm_or_si128(x, v_independent1), v_key1),
        _mm_cmpeq_epi8(_mm_or_si128(x, v_independent2), v_key2),
    )) as u32
}

#[target_feature(enable = "sse2")]
pub unsafe fn escape(feed: &str, buffer: &mut Buffer) { unsafe {
    debug_assert!(feed.len() >= 16);

    let len = feed.len();
    let mut ptr = feed.as_ptr();
    let end_ptr = ptr.add(len);

    // Clean blocks are stored as soon as they are scanned, instead of being copied
    // in a second pass. Room is kept for the rest of the input, and extended
    // before a block is escaped.
    buffer.reserve_small(len);
    let mut dst = buffer.as_mut_ptr().add(buffer.len());

    while end_ptr as usize - ptr as usize >= 2 * VECTOR_BYTES {
        let lo_v = _mm_loadu_si128(ptr as *const __m128i);
        let hi_v = _mm_loadu_si128(ptr.add(VECTOR_BYTES) as *const __m128i);
        let lo = maskgen(lo_v);
        let hi = maskgen(hi_v);
        if (lo | hi) == 0 {
            _mm_storeu_si128(dst as *mut __m128i, lo_v);
            _mm_storeu_si128(dst.add(VECTOR_BYTES) as *mut __m128i, hi_v);
            dst = dst.add(2 * VECTOR_BYTES);
        } else {
            let room = end_ptr as usize - ptr as usize + BLOCK_RESERVE;
            dst = commit_and_reserve(buffer, dst, room);
            dst = escape_block(lo | (hi << VECTOR_BYTES), ptr, 2 * VECTOR_BYTES, dst);
        }
        ptr = ptr.add(2 * VECTOR_BYTES);
    }

    if end_ptr as usize - ptr as usize >= VECTOR_BYTES {
        let v = _mm_loadu_si128(ptr as *const __m128i);
        let mask = maskgen(v);
        if mask == 0 {
            _mm_storeu_si128(dst as *mut __m128i, v);
            dst = dst.add(VECTOR_BYTES);
        } else {
            let room = end_ptr as usize - ptr as usize + BLOCK_RESERVE;
            dst = commit_and_reserve(buffer, dst, room);
            dst = escape_block(mask, ptr, VECTOR_BYTES, dst);
        }
        ptr = ptr.add(VECTOR_BYTES);
    }

    let rest = end_ptr as usize - ptr as usize;
    debug_assert!(rest < VECTOR_BYTES);
    if rest > 0 {
        // the last vector of the input, shifted so that bit 0 stands for `ptr`
        let backs = VECTOR_BYTES - rest;
        let mask = maskgen(_mm_loadu_si128(ptr.sub(backs) as *const __m128i)) >> backs;
        if mask == 0 {
            memcpy_16(ptr, dst, rest);
            dst = dst.add(rest);
        } else {
            dst = commit_and_reserve(buffer, dst, rest + BLOCK_RESERVE);
            dst = escape_block(mask, ptr, rest, dst);
        }
    }

    commit(buffer, dst);
}}

/// Escapes the `len` bytes at `src` into `dst`, returning the end of the output
///
/// Bit `i` of `mask` flags `src + i` as possibly needing escaping, and every byte
/// that needs it must be flagged.
///
/// # Safety
///
/// `len` must be at most 32, and `dst` valid for writing `len * 6 + 2` bytes.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn escape_block(
    mut mask: u32,
    src: *const u8,
    len: usize,
    mut dst: *mut u8,
) -> *mut u8 { unsafe {
    debug_assert!(len <= 32);

    let mut start = src;
    while mask != 0 {
        let p = src.add(mask.trailing_zeros() as usize);
        mask &= mask - 1;
        let c = *ESCAPE_LUT.get_unchecked(*p as usize) as usize;
        if c < ESCAPED_LEN {
            let run = p as usize - start as usize;
            copy_32(start, dst, run);
            dst = dst.add(run);
            ptr::write_unaligned(dst as *mut [u8; 8], *ESCAPED_PADDED.get_unchecked(c));
            dst = dst.add(ESCAPED.get_unchecked(c).len());
            start = p.add(1);
        }
    }

    let run = src.add(len) as usize - start as usize;
    copy_32(start, dst, run);
    dst.add(run)
}}

/// Copies up to 32 bytes with at most two loads and stores, which may overlap
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn copy_32(src: *const u8, dst: *mut u8, len: usize) { unsafe {
    debug_assert!(len <= 32);
    if len <= 16 {
        memcpy_16(src, dst, len);
    } else {
        let head = _mm_loadu_si128(src as *const __m128i);
        let tail = _mm_loadu_si128(src.add(len - 16) as *const __m128i);
        _mm_storeu_si128(dst as *mut __m128i, head);
        _mm_storeu_si128(dst.add(len - 16) as *mut __m128i, tail);
    }
}}
