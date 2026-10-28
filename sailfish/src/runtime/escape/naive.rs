use std::ptr;
use std::slice;

use super::super::utils::memcpy_16;
use super::super::Buffer;
use super::fallback::key_bits;
use super::{ESCAPED, ESCAPED_LEN, ESCAPE_LUT};

// the reference implementation, used under Miri and by the tests
#[cfg_attr(not(miri), allow(dead_code))]
#[inline]
pub(super) unsafe fn escape(
    buffer: &mut Buffer,
    mut start_ptr: *const u8,
    ptr: *const u8,
    end_ptr: *const u8,
) { unsafe {
    start_ptr = proceed(buffer, start_ptr, ptr, end_ptr);

    if likely!(end_ptr > start_ptr) {
        let slc = slice::from_raw_parts(start_ptr, end_ptr as usize - start_ptr as usize);
        buffer.push_str(std::str::from_utf8_unchecked(slc));
    }
}}

#[cfg_attr(not(miri), allow(dead_code))]
#[inline]
pub(super) unsafe fn proceed(
    buffer: &mut Buffer,
    mut start_ptr: *const u8,
    mut ptr: *const u8,
    end_ptr: *const u8,
) -> *const u8 { unsafe {
    while ptr < end_ptr {
        debug_assert!(start_ptr <= ptr);
        let idx = ESCAPE_LUT[*ptr as usize] as usize;
        debug_assert!(idx <= 9);
        if likely!(idx >= ESCAPED_LEN) {
            ptr = ptr.add(1);
        } else {
            if ptr > start_ptr {
                let slc =
                    slice::from_raw_parts(start_ptr, ptr as usize - start_ptr as usize);
                buffer.push_str(std::str::from_utf8_unchecked(slc));
            }
            push_escaped_str(ESCAPED.get_unchecked(idx), buffer);
            start_ptr = ptr.add(1);
            ptr = ptr.add(1);
        }
    }

    debug_assert_eq!(ptr, end_ptr);
    debug_assert!(start_ptr <= ptr);
    start_ptr
}}

/// Every byte's escaped form, padded to 8 bytes, with its length in the last byte
static ESCAPE_WORDS: [[u8; 8]; 256] = {
    let mut table = [[0u8; 8]; 256];
    let mut c = 0;
    while c < 256 {
        table[c][0] = c as u8;
        table[c][7] = 1;
        c += 1;
    }

    let mut i = 0;
    while i < ESCAPED_LEN {
        let key = b"\"&'<>"[i] as usize;
        let escaped = ESCAPED[i].as_bytes();
        let mut j = 0;
        while j < escaped.len() {
            table[key][j] = escaped[j];
            j += 1;
        }
        table[key][7] = escaped.len() as u8;
        i += 1;
    }
    table
};

/// Returns the index of the first byte of `feed` that may need escaping, or
/// `feed.len()` if there is none
///
/// Short values usually need no escaping, so this checks whole words instead of
/// single bytes. The two loads overlap when `len < 16`, and the lowest flagged
/// byte is exact, though it may be a `#`. Below 4 bytes, the result is only
/// exact for contents without escapable bytes, and is 0 otherwise.
#[inline]
pub(super) fn first_escape_small(feed: &str) -> usize {
    debug_assert!(feed.len() < 16);

    let len = feed.len();
    let src = feed.as_ptr();
    // SAFETY: every load is within `feed`
    unsafe {
        if len >= 8 {
            let head = key_bits(u64::from_le(ptr::read_unaligned(src as *const u64)));
            let tail = key_bits(u64::from_le(ptr::read_unaligned(
                src.add(len - 8) as *const u64,
            )));
            if head != 0 {
                head.trailing_zeros() as usize / 8
            } else if tail != 0 {
                len - 8 + tail.trailing_zeros() as usize / 8
            } else {
                len
            }
        } else if len >= 4 {
            let head = u32::from_le(ptr::read_unaligned(src as *const u32)) as u64;
            let tail = u32::from_le(ptr::read_unaligned(src.add(len - 4) as *const u32));
            // zero bytes never match, so the words can be packed together
            let bits = key_bits(head | ((tail as u64) << 32));
            match bits.trailing_zeros() as usize / 8 {
                i @ 0..4 => i,
                i @ 4..8 => i + len - 8,
                _ => len,
            }
        } else if len > 0 {
            // bytes 0, len / 2 and len - 1 cover every position. Escaping
            // starts from the beginning if any of them needs it.
            let lut = |i: usize| *ESCAPE_LUT.get_unchecked(*src.add(i) as usize);
            if lut(0) & lut(len / 2) & lut(len - 1) == 9 { len } else { 0 }
        } else {
            0
        }
    }
}

/// Escapes `feed` into `buf`, returning the number of bytes written
///
/// # Safety
///
/// - `feed` must be shorter than 16 bytes, and no byte before `first` may need
///   escaping
/// - `buf` must be valid for writing `feed.len() * 6 + 2` bytes
pub(super) unsafe fn escape_small(
    feed: &str,
    mut buf: *mut u8,
    first: usize,
) -> usize { unsafe {
    debug_assert!(feed.len() < 16);
    debug_assert!(first <= feed.len());

    let len = feed.len();
    let src = feed.as_ptr();
    let buf_begin = buf;
    memcpy_16(src, buf, first);
    if first == len {
        return len;
    }
    buf = escape_bytes(src.add(first), len - first, buf.add(first));
    buf as usize - buf_begin as usize
}}

/// Escapes the `len` bytes at `src` into `dst` one by one without branches,
/// returning the end of the output
///
/// # Safety
///
/// `dst` must be valid for writing `len * 6 + 2` bytes.
#[inline]
pub(super) unsafe fn escape_bytes(
    src: *const u8,
    len: usize,
    mut dst: *mut u8,
) -> *mut u8 { unsafe {
    // Each word holds the byte's escaped form, with its length in the last byte.
    let end = src.add(len);
    let mut ptr = src;
    while ptr < end {
        let word = ESCAPE_WORDS.get_unchecked(*ptr as usize);
        ptr::write_unaligned(dst as *mut [u8; 8], *word);
        dst = dst.add(word[7] as usize);
        ptr = ptr.add(1);
    }
    dst
}}

/// Escapes the 8 bytes loaded into `bytes` into `dst` without branches, returning
/// the end of the output
///
/// All bytes are looked up before any is stored. Otherwise, should the CPU
/// predict that a lookup depends on the store before it, every byte would wait
/// for the previous one, which makes escaping long contents up to 3 times slower.
///
/// # Safety
///
/// `dst` must be valid for writing 50 bytes.
#[inline]
pub(super) unsafe fn escape_word(bytes: u64, mut dst: *mut u8) -> *mut u8 { unsafe {
    let bytes = u64::from_le(bytes);
    let words: [[u8; 8]; 8] = std::array::from_fn(|i| {
        *ESCAPE_WORDS.get_unchecked((bytes >> (i * 8)) as u8 as usize)
    });
    for word in words {
        ptr::write_unaligned(dst as *mut [u8; 8], word);
        dst = dst.add(word[7] as usize);
    }
    dst
}}

/// Sets the end of `buffer`'s contents to `dst`, which must be within its capacity
/// and not before its current end
#[inline]
pub(super) unsafe fn commit(buffer: &mut Buffer, dst: *mut u8) { unsafe {
    let begin = buffer.as_mut_ptr().add(buffer.len());
    debug_assert!(dst >= begin);
    buffer.advance(dst as usize - begin as usize);
}}

/// Commits the output written up to `dst`, and makes room for `size` more bytes
///
/// Returns the position to write at, which moves if the buffer grows.
#[inline]
pub(super) unsafe fn commit_and_reserve(
    buffer: &mut Buffer,
    dst: *mut u8,
    size: usize,
) -> *mut u8 { unsafe {
    commit(buffer, dst);
    buffer.reserve_small(size);
    buffer.as_mut_ptr().add(buffer.len())
}}

#[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
#[cfg_attr(not(miri), allow(dead_code))]
#[inline]
pub(super) unsafe fn push_escaped_str(value: &str, buffer: &mut Buffer) { unsafe {
    buffer.reserve_small(value.len());

    let src = value.as_ptr();
    let dst = buffer.as_mut_ptr().add(buffer.len());

    // memcpy
    let offset = value.len() - 4;
    let t2 = ptr::read_unaligned(src.add(offset) as *const u32);
    let t1 = ptr::read_unaligned(src as *const u32);
    ptr::write_unaligned(dst.add(offset) as *mut u32, t2);
    ptr::write_unaligned(dst as *mut u32, t1);

    buffer._set_len(buffer.len() + value.len());
}}

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
#[inline]
pub(super) unsafe fn push_escaped_str(value: &str, buffer: &mut Buffer) {
    buffer.push_str(value);
}
