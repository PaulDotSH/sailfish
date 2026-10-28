use std::fmt;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sailfish::TemplateSimple;
use sailfish::runtime::escape::{escape_to_buf, escape_to_string};
use sailfish::runtime::filter::{disp, lower, truncate, upper};
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

/// ~1 KB of prose with an escapable byte roughly every 100 bytes
fn long_sparse() -> String {
    "Sailfish renders templates quickly, and it doesn't allocate needlessly; numbers like 3 < 4 hold. "
        .repeat(10)
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

fn bench_escape(c: &mut Criterion) {
    let mut group = c.benchmark_group("escape");
    let mut buf = Buffer::with_capacity(8192);

    let mut cases = inputs();
    cases.push(("medium_plain", "x".repeat(48)));
    cases.push((
        "medium_escape",
        "<a href=\"/u/1\">Tom & Jerry</a>".to_owned(),
    ));
    cases.push(("long_sparse", long_sparse()));

    for (name, input) in &cases {
        group.bench_with_input(BenchmarkId::from_parameter(name), input, |bench, s| {
            bench.iter(|| {
                buf.clear();
                escape_to_buf(black_box(s), &mut buf);
            })
        });
    }

    group.finish();
}

fn bench_truncate(c: &mut Criterion) {
    let mut group = c.benchmark_group("truncate");
    let mut buf = Buffer::with_capacity(4096);
    let long = long_plain();

    for (name, input) in [("short", SHORT_PLAIN), ("long", long.as_str())] {
        group.bench_with_input(BenchmarkId::from_parameter(name), input, |bench, s| {
            bench.iter(|| {
                buf.clear();
                truncate(black_box(s), 100)
                    .render_escaped(&mut buf)
                    .unwrap();
            })
        });
    }

    group.finish();
}

#[derive(TemplateSimple)]
#[template(source = "<table><% for row in table { %>
<tr><% for col in row { %><td><%= col %></td><% } %></tr><% } %>
</table>")]
struct BigTable<'a> {
    table: &'a [Vec<usize>],
}

struct Team {
    name: String,
    score: u8,
}

#[derive(TemplateSimple)]
#[template(source = "<html>
  <head>
    <title><%= year %></title>
  </head>
  <body>
    <h1>CSL <%= year %></h1>
    <ul>
    <% for (i, team) in teams.iter().enumerate() { %>
      <li class=\"<% if i == 0 { %>champion<% } %>\">
      <b><%= team.name %></b>: <%= team.score %>
      </li>
    <% } %>
    </ul>
  </body>
</html>")]
struct Teams<'a> {
    year: u16,
    teams: &'a [Team],
}

struct Comment {
    author: String,
    body: String,
    votes: i32,
}

#[derive(TemplateSimple)]
#[template(source = "<section><% for c in comments { %>
<article><h3><%= c.author %></h3><p><%= c.body %></p><span><%= c.votes %></span></article><% } %>
</section>")]
struct Comments<'a> {
    comments: &'a [Comment],
}

fn bench_templates(c: &mut Criterion) {
    let mut group = c.benchmark_group("template");

    let table: Vec<Vec<usize>> = (0..100).map(|_| (0..100).collect()).collect();
    group.bench_function("big_table", |bench| {
        bench.iter(|| {
            BigTable {
                table: black_box(&table),
            }
            .render_once()
            .unwrap()
        })
    });

    let teams = [
        ("Jiangsu", 43),
        ("Beijing", 27),
        ("Guangzhou", 22),
        ("Shandong", 12),
    ]
    .map(|(name, score)| Team {
        name: name.to_owned(),
        score,
    });
    group.bench_function("teams", |bench| {
        bench.iter(|| {
            Teams {
                year: 2015,
                teams: black_box(&teams),
            }
            .render_once()
            .unwrap()
        })
    });

    let comments: Vec<Comment> = (0..50)
        .map(|i| Comment {
            author: format!("user{i}"),
            body: if i % 3 == 0 {
                "I think <b>this</b> is \"great\" & I'd use it again.".to_owned()
            } else {
                "Thanks for sharing, this was a very helpful explanation of the topic."
                    .to_owned()
            },
            votes: i * 7 - 100,
        })
        .collect();
    group.bench_function("comments", |bench| {
        bench.iter(|| {
            Comments {
                comments: black_box(&comments),
            }
            .render_once()
            .unwrap()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_escape,
    bench_truncate,
    bench_templates,
    bench_default_render_escaped,
    bench_case_filters,
    bench_string_conversion
);
criterion_main!(benches);
