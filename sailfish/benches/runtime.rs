use std::fmt;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sailfish::runtime::escape::escape_to_string;
use sailfish::runtime::filter::{disp, lower, upper};
use sailfish::runtime::{Buffer, Render, RenderError};

const SHORT_PLAIN: &str = "Hello, world";
const SHORT_ESCAPE: &str = "a < b & c";
const NON_ASCII: &str = "Grüße aus Köln, ὈΔΥΣΣΕΎΣ";

fn long_plain() -> String {
    "The quick brown fox jumps over the lazy dog. ".repeat(24)
}

fn long_escape() -> String {
    "<p class=\"note\">Tom & Jerry's show</p> ".repeat(24)
}

/// A user type that only implements `render`, so `render_escaped` uses the
/// trait's default implementation.
struct Custom<'a>(&'a str);

impl Render for Custom<'_> {
    fn render(&self, b: &mut Buffer) -> Result<(), RenderError> {
        b.push_str(self.0);
        Ok(())
    }
}

struct Point(i32, i32);

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

fn inputs() -> Vec<(&'static str, String)> {
    vec![
        ("short_plain", SHORT_PLAIN.to_owned()),
        ("short_escape", SHORT_ESCAPE.to_owned()),
        ("long_plain", long_plain()),
        ("long_escape", long_escape()),
    ]
}

fn bench_default_render_escaped(c: &mut Criterion) {
    let mut group = c.benchmark_group("default_render_escaped");
    let mut buf = Buffer::with_capacity(4096);

    for (name, input) in inputs() {
        group.bench_with_input(BenchmarkId::from_parameter(name), &input, |bench, s| {
            bench.iter(|| {
                buf.clear();
                Custom(black_box(s)).render_escaped(&mut buf).unwrap();
            })
        });
    }

    group.bench_function("disp_point", |bench| {
        let p = Point(-1234, 5678);
        bench.iter(|| {
            buf.clear();
            disp(black_box(&p)).render_escaped(&mut buf).unwrap();
        })
    });

    group.finish();
}

fn bench_case_filters(c: &mut Criterion) {
    let mut group = c.benchmark_group("case_filters");
    let mut buf = Buffer::with_capacity(4096);

    let mut cases = inputs();
    cases.push(("non_ascii", NON_ASCII.to_owned()));

    for (name, input) in &cases {
        let s = input.as_str();
        group.bench_with_input(BenchmarkId::new("upper", name), s, |bench, s| {
            bench.iter(|| {
                buf.clear();
                upper(black_box(s)).render(&mut buf).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("upper_escaped", name), s, |bench, s| {
            bench.iter(|| {
                buf.clear();
                upper(black_box(s)).render_escaped(&mut buf).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("lower", name), s, |bench, s| {
            bench.iter(|| {
                buf.clear();
                lower(black_box(s)).render(&mut buf).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("lower_escaped", name), s, |bench, s| {
            bench.iter(|| {
                buf.clear();
                lower(black_box(s)).render_escaped(&mut buf).unwrap();
            })
        });
    }

    group.finish();
}

fn bench_string_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_conversion");

    // `escape_to_string` into a reused `String` with spare capacity.
    for (name, input) in inputs() {
        group.bench_with_input(
            BenchmarkId::new("escape_to_string", name),
            &input,
            |bench, s| {
                let mut out = String::with_capacity(8192);
                bench.iter(|| {
                    out.clear();
                    escape_to_string(black_box(s), &mut out);
                })
            },
        );
    }

    group.bench_function("buffer_from_string_roundtrip", |bench| {
        let mut s = String::with_capacity(4096);
        bench.iter(|| {
            s.clear();
            s.push_str("abc");
            let mut buf = Buffer::from(std::mem::take(&mut s));
            buf.push_str(black_box(SHORT_PLAIN));
            s = buf.into_string();
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_default_render_escaped,
    bench_case_filters,
    bench_string_conversion
);
criterion_main!(benches);
