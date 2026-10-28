#![allow(clippy::cast_ptr_alignment)]

use std::ptr;

use super::super::Buffer;
use super::super::utils::memcpy_16;
use super::naive;

/// Flags the bytes of `x` that may need escaping by setting their high bit
///
/// The lowest flagged byte is always exact, but bytes above it may be flagged
/// spuriously. `#` is flagged too.
#[inline]
pub(super) fn key_bits(x: u64) -> u64 {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    let zero_bits = |x: u64| x.wrapping_sub(LO) & !x & HI;

    zero_bits((x | 0x0505_0505_0505_0505) ^ 0x2727_2727_2727_2727)
        | zero_bits((x | 0x0202_0202_0202_0202) ^ 0x3e3e_3e3e_3e3e_3e3e)
}

/// Room needed to escape a block of 16 bytes, besides the rest of the input: each
/// byte expands to at most 6 bytes, and the last entity may be written 2 bytes
/// past its end.
const BLOCK_RESERVE: usize = 16 * 6 + 2;

#[inline]
pub unsafe fn escape(feed: &str, buffer: &mut Buffer) { unsafe {
    debug_assert!(feed.len() >= 16);

    let len = feed.len();
    let mut ptr = feed.as_ptr();
    let end_ptr = ptr.add(len);
    let load = |p: *const u8| ptr::read_unaligned(p as *const u64);
    let flags = |x: u64| key_bits(u64::from_le(x));

    // Clean blocks are stored as soon as they are checked, instead of being copied
    // in a second pass. Room is kept for the rest of the input, and extended
    // before a block is escaped.
    buffer.reserve_small(len);
    let mut dst = buffer.as_mut_ptr().add(buffer.len());

    while end_ptr as usize - ptr as usize >= 16 {
        let lo = load(ptr);
        let hi = load(ptr.add(8));
        if flags(lo) | flags(hi) == 0 {
            ptr::write_unaligned(dst as *mut u64, lo);
            ptr::write_unaligned(dst.add(8) as *mut u64, hi);
            dst = dst.add(16);
        } else {
            dst = naive::commit_and_reserve(
                buffer,
                dst,
                end_ptr as usize - ptr as usize + BLOCK_RESERVE,
            );
            dst = naive::escape_word(lo, dst);
            dst = naive::escape_word(hi, dst);
        }
        ptr = ptr.add(16);
    }

    let rest = end_ptr as usize - ptr as usize;
    if rest > 0 {
        // Check the words ending at the end of the input. Dropping the flags of
        // bytes already done keeps every flag of the rest, if not only those.
        let tail = flags(load(end_ptr.sub(8)));
        let found = if rest >= 8 {
            flags(load(ptr)) | tail
        } else {
            tail >> (8 * (8 - rest))
        };
        if found == 0 {
            memcpy_16(ptr, dst, rest);
            dst = dst.add(rest);
        } else {
            dst = naive::commit_and_reserve(buffer, dst, rest + BLOCK_RESERVE);
            dst = naive::escape_bytes(ptr, rest, dst);
        }
    }

    naive::commit(buffer, dst);
}}
