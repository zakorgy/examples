extern crate vulkano_shaders;

fn main() {
    // building the shaders used in the examples
    vulkano_shaders::build_glsl_shaders([
        ("src/shader/image_vs.glsl", vulkano_shaders::ShaderType::Vertex),
		("src/shader/image_fs.glsl", vulkano_shaders::ShaderType::Fragment),
    ].iter().cloned());
}