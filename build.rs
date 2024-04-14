use cmake;

fn main() {
    cmake::build("kernel");
    println!("cargo::rerun-if-changed=kernel/*.metal");
}
