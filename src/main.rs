extern crate cgmath;
extern crate euclid;
extern crate gleam;
extern crate glutin;
extern crate image;
extern crate lodepng;
extern crate offscreen_gl_context;
#[macro_use]
extern crate vulkano;
extern crate vulkano_win;
extern crate winit;

use euclid::Size2D;
use gleam::gl;
use offscreen_gl_context::{ColorAttachmentType, GLContext, GLContextAttributes, NativeGLContext};
use std::time::Duration;
use vulkano_win::VkSurfaceBuild;
use std::cell::Cell;

struct GlHandles {
    pub vao: Cell<gl::GLuint>,
    pub vbos: Cell<[gl::GLuint; 2]>,
    pub program: Cell<gl::GLuint>,
    pub scale: Cell<f32>,
}

impl GlHandles {
    fn new() -> GlHandles {
        GlHandles {
            vao: Cell::new(0 as gl::GLuint),
            vbos: Cell::new([0,0]),
            program: Cell::new(0 as gl::GLuint),
            scale: Cell::new(0.0),
        }
    }
}

const CLEAR_COLOR: (f32, f32, f32, f32) = (0.0, 0.2, 0.3, 1.0);
const WIN_WIDTH: i32 = 256;
const WIN_HEIGTH: i32 = 256;
const VS_SHADER: &'static str = "
    #version 150 core

    in vec2 a_Pos;
    in vec3 a_Color;
    uniform mat4 scale;
    out vec4 v_Color;

    void main() {
        v_Color = vec4(a_Color, 1.0);
        gl_Position = scale * vec4(0.5 * a_Pos, 0.0, 1.0);
    }
";

const FS_SHADER: &'static str = "
    #version 150 core

    in vec4 v_Color;
    out vec4 Target0;

    void main() {
        Target0 = v_Color;
    }
";

const VERTICES: &'static [[f32;2];3] = &[
        [-1.0, -0.57],
        [ 1.0, -0.57],
        [ 0.0,  1.15]
    ];

const COLORS: &'static [[f32;3];3] = &[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ];

fn add_shader(program: gl::GLuint, src: &str, ty: gl::GLenum) {
    let id = unsafe { gl::CreateShader(ty) };
    if id == (0 as gl::GLuint) {
        panic!("Failed to create shader type: {:?}", ty);
    }
    let mut source = Vec::new();
    source.extend_from_slice(src.as_bytes());
    gl::shader_source(id, &[&source[..]]);
    gl::compile_shader(id);
    let log = gl::get_shader_info_log(id);
    if gl::get_shader_iv(id, gl::COMPILE_STATUS) == (0 as gl::GLint) {
        panic!("Failed to compile shader:\n{}", log);
    } else {
        if !log.is_empty() {
            println!("Warnings detected on shader:\n{}", log);
        }
        gl::attach_shader(program, id);
    }
}

fn compile_shaders(handles: &GlHandles) {
    handles.program.set(gl::create_program());
    if handles.program.get() == (0 as gl::GLuint) {
        panic!("Failed to create shader program");
    }

    add_shader(handles.program.get(), VS_SHADER, gl::VERTEX_SHADER);
    add_shader(handles.program.get(), FS_SHADER, gl::FRAGMENT_SHADER);

    gl::link_program(handles.program.get());
    if gl::get_program_iv(handles.program.get(), gl::LINK_STATUS) == (0 as gl::GLint) {
        let error_log = gl::get_program_info_log(handles.program.get());
        panic!("Failed to link shader program: \n{}", error_log);
    }

    gl::validate_program(handles.program.get());
    if gl::get_program_iv(handles.program.get(), gl::VALIDATE_STATUS) == (0 as gl::GLint) {
        let error_log = gl::get_program_info_log(handles.program.get());
        panic!("Failed to validate shader program: \n{}", error_log);
    }

    gl::use_program(handles.program.get());
}

fn draw(handles: &GlHandles) {
    gl::clear_color(CLEAR_COLOR.0, CLEAR_COLOR.1, CLEAR_COLOR.2, CLEAR_COLOR.3);
    gl::clear(gl::COLOR_BUFFER_BIT);

    gl::bind_buffer(gl::ARRAY_BUFFER, handles.vbos.get()[0]);
    gl::buffer_data(gl::ARRAY_BUFFER, VERTICES, gl::STATIC_DRAW);
    gl::vertex_attrib_pointer(0 as gl::GLuint,
                              2,
                              gl::FLOAT,
                              false,
                              0 as gl::GLint,
                              0);
    gl::enable_vertex_attrib_array(0 as gl::GLuint);

    gl::bind_buffer(gl::ARRAY_BUFFER, handles.vbos.get()[1]);
    gl::buffer_data(gl::ARRAY_BUFFER, COLORS, gl::STATIC_DRAW);
    gl::vertex_attrib_pointer(1 as gl::GLuint,
                              3,
                              gl::FLOAT,
                              false,
                              0 as gl::GLint,
                              0);
    gl::enable_vertex_attrib_array(1 as gl::GLuint);

    handles.scale.set(handles.scale.get() + 0.01);

    let scale = handles.scale.get();
    let rot_matrix = [
        scale.cos(), -1.0 * scale.sin(), 0.0, 0.0,
        scale.sin(),        scale.cos(), 0.0, 0.0,
                0.0,                0.0, 1.0, 0.0,
                0.0,                0.0, 0.0, 1.0
        ];

    let scale_pos = gl::get_uniform_location(handles.program.get(), "scale");
    // gl::uniform_1f(scale, handles.scale.get().sin());
    gl::uniform_matrix_4fv(scale_pos, false, &rot_matrix);

    gl::draw_arrays(gl::TRIANGLES, 0, 3);
}

pub fn main() {
    gl::load_with(|s| GLContext::<NativeGLContext>::get_proc_address(s) as *const _);
    let offscreen_ctx = GLContext::<NativeGLContext>::new(Size2D::new(256, 256),
                                                          GLContextAttributes::default(),
                                                          ColorAttachmentType::Renderbuffer,
                                                          None).unwrap();
    offscreen_ctx.make_current().unwrap();

    /*let window = glutin::WindowBuilder::new()
        .with_title("Triangle example".to_string())
        .with_dimensions(WIN_WIDTH as u32, WIN_HEIGTH as u32)
        .with_vsync()
        .build()
        .unwrap();

    unsafe { window.make_current().unwrap() };
    gl::load_with(|s| window.get_proc_address(s) as *const _);*/


    let handles: GlHandles = GlHandles::new();
    // Create VAO and VBOs
    handles.vao.set(gl::gen_vertex_arrays(1)[0]);
    gl::bind_vertex_array(handles.vao.get());
    let buffer_ids = gl::gen_buffers(2);
    handles.vbos.set([buffer_ids[0], buffer_ids[1]]);

    compile_shaders(&handles);

    // Create FBO and bind it
    let fbo = gl::gen_framebuffers(1)[0];
    gl::bind_framebuffer(gl::FRAMEBUFFER, fbo);

    // Create renderbuffer for color data
    let color_render_buffer = gl::gen_renderbuffers(1)[0];
    gl::bind_renderbuffer(gl::RENDERBUFFER, color_render_buffer);
    gl::renderbuffer_storage(gl::RENDERBUFFER, gl::RGBA, WIN_WIDTH, WIN_HEIGTH);
    gl::framebuffer_renderbuffer(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::RENDERBUFFER, color_render_buffer);

    if gl::check_frame_buffer_status(gl::FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE {
        panic!("Something went wrong :/.");
    }

    // Draw in the FBO and read the pixel data from it
    draw(&handles);
    let pixel_data = gl::read_pixels(0, 0, WIN_WIDTH, WIN_HEIGTH, gl::RGBA, gl::UNSIGNED_BYTE);

    // Create image from pixeldata, this is just for check
    /*let path = &std::path::Path::new("write_test.png");
    if let Err(e) = lodepng::encode_file(path,
                                         &pixel_data,
                                         WIN_WIDTH as usize,
                                         WIN_HEIGTH as usize,
                                         lodepng::LCT_RGBA,
                                         8) {
            panic!("failed to write png: {:?}", e);
    }



    'main: loop {
        for event in window.poll_events() {
            match event {
                glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Escape)) |
                glutin::Event::Closed => break 'main,
                _ => {},
            }
        }
        draw(&handles);
        window.swap_buffers().unwrap();
    }*/


    let extensions = vulkano_win::required_extensions();
    let instance = vulkano::instance::Instance::new(None, &extensions, &[]).expect("failed to create instance");

    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
                            .next().expect("no device available");
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let window = winit::WindowBuilder::new().build_vk_surface(&instance).unwrap();

    let queue = physical.queue_families().find(|q| q.supports_graphics() &&
                                                   window.surface().is_supported(q).unwrap_or(false))
                                                .expect("couldn't find a graphical queue family");

    let device_ext = vulkano::device::DeviceExtensions {
        khr_swapchain: true,
        .. vulkano::device::DeviceExtensions::none()
    };
    let (device, mut queues) = vulkano::device::Device::new(&physical, physical.supported_features(),
                                                            &device_ext, [(queue, 0.5)].iter().cloned())
                               .expect("failed to create device");
    let queue = queues.next().unwrap();

    let (swapchain, images) = {
        let caps = window.surface().get_capabilities(&physical).expect("failed to get surface capabilities");

        let dimensions = caps.current_extent.unwrap_or([WIN_WIDTH as u32, WIN_HEIGTH as u32]);
        let present = caps.present_modes.iter().next().unwrap();
        let usage = caps.supported_usage_flags;

        vulkano::swapchain::Swapchain::new(&device, &window.surface(), caps.min_image_count,
                                           vulkano::format::B8G8R8A8Srgb, dimensions, 1,
                                           &usage, &queue, vulkano::swapchain::SurfaceTransform::Identity,
                                           vulkano::swapchain::CompositeAlpha::Opaque,
                                           present, true, None).expect("failed to create swapchain")
    };


    #[derive(Debug, Clone)]
    struct Vertex { position: [f32; 2] }
    impl_vertex!(Vertex, position);

    let vertex_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer::<[Vertex]>
                               ::from_iter(&device, &vulkano::buffer::BufferUsage::all(),
                                       Some(queue.family()), [
                                           Vertex { position: [-0.5, -0.5 ] },
                                           Vertex { position: [-0.5,  0.5 ] },
                                           Vertex { position: [ 0.5, -0.5 ] },
                                           Vertex { position: [ 0.5,  0.5 ] },
                                       ].iter().cloned()).expect("failed to create buffer");

    mod vs { include!{concat!(env!("OUT_DIR"), "/shaders/src/shader/image_vs.glsl")} }
    let vs = vs::Shader::load(&device).expect("failed to create shader module");
    mod fs { include!{concat!(env!("OUT_DIR"), "/shaders/src/shader/image_fs.glsl")} }
    let fs = fs::Shader::load(&device).expect("failed to create shader module");

    mod renderpass {
        single_pass_renderpass!{
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: ::vulkano::format::B8G8R8A8Srgb,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        }
    }

    let renderpass = renderpass::CustomRenderPass::new(&device, &renderpass::Formats {
        color: (vulkano::format::B8G8R8A8Srgb, 1)
    }).unwrap();

    let texture = vulkano::image::immutable::ImmutableImage::new(&device, vulkano::image::Dimensions::Dim2d { width: WIN_WIDTH as u32, height: WIN_HEIGTH as u32 },
                                                                 vulkano::format::R8G8B8A8Unorm, Some(queue.family())).unwrap();

    let sampler = vulkano::sampler::Sampler::new(&device, vulkano::sampler::Filter::Linear,
                                                 vulkano::sampler::Filter::Linear, vulkano::sampler::MipmapMode::Nearest,
                                                 vulkano::sampler::SamplerAddressMode::Repeat,
                                                 vulkano::sampler::SamplerAddressMode::Repeat,
                                                 vulkano::sampler::SamplerAddressMode::Repeat,
                                                 0.0, 1.0, 0.0, 0.0).unwrap();

    let descriptor_pool = vulkano::descriptor::descriptor_set::DescriptorPool::new(&device);
    mod pipeline_layout {
        pipeline_layout!{
            set0: {
                tex: CombinedImageSampler
            }
        }
    }

    let pipeline_layout = pipeline_layout::CustomPipeline::new(&device).unwrap();
    let set = pipeline_layout::set0::Set::new(&descriptor_pool, &pipeline_layout, &pipeline_layout::set0::Descriptors {
        tex: (&sampler, &texture)
    });


    let pipeline = vulkano::pipeline::GraphicsPipeline::new(&device, vulkano::pipeline::GraphicsPipelineParams {
        vertex_input: vulkano::pipeline::vertex::SingleBufferDefinition::new(),
        vertex_shader: vs.main_entry_point(),
        input_assembly: vulkano::pipeline::input_assembly::InputAssembly {
            topology: vulkano::pipeline::input_assembly::PrimitiveTopology::TriangleStrip,
            primitive_restart_enable: false,
        },
        tessellation: None,
        geometry_shader: None,
        viewport: vulkano::pipeline::viewport::ViewportsState::Fixed {
            data: vec![(
                vulkano::pipeline::viewport::Viewport {
                    origin: [0.0, 0.0],
                    depth_range: 0.0 .. 1.0,
                    dimensions: [images[0].dimensions()[0] as f32, images[0].dimensions()[1] as f32],
                },
                vulkano::pipeline::viewport::Scissor::irrelevant()
            )],
        },
        raster: Default::default(),
        multisample: vulkano::pipeline::multisample::Multisample::disabled(),
        fragment_shader: fs.main_entry_point(),
        depth_stencil: vulkano::pipeline::depth_stencil::DepthStencil::disabled(),
        blend: vulkano::pipeline::blend::Blend::pass_through(),
        layout: &pipeline_layout,
        render_pass: vulkano::framebuffer::Subpass::from(&renderpass, 0).unwrap(),
    }).unwrap();

    let framebuffers = images.iter().map(|image| {
        let attachments = renderpass::AList {
            color: &image,
        };

        vulkano::framebuffer::Framebuffer::new(&renderpass, [images[0].dimensions()[0], images[0].dimensions()[1], 1], attachments).unwrap()
    }).collect::<Vec<_>>();

    let pixel_buffer = {
        let image_data_chunks = pixel_data.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]);

        vulkano::buffer::cpu_access::CpuAccessibleBuffer::<[[u8;4]]>
            ::from_iter(&device, &vulkano::buffer::BufferUsage::all(),
                        Some(queue.family()), image_data_chunks)
                        .expect("failed to create buffer")

    };

    let command_buffers = framebuffers.iter().map(|framebuffer| {
        vulkano::command_buffer::PrimaryCommandBufferBuilder::new(&device, queue.family())
            .copy_buffer_to_color_image(&pixel_buffer, &texture, 0, 0 .. 1, [0, 0, 0],
                                        [texture.dimensions().width(), texture.dimensions().height(), 1])
            //.clear_color_image(&texture, [0.0, 1.0, 0.0, 1.0])
            .draw_inline(&renderpass, &framebuffer, renderpass::ClearValues {
                color: [0.0, 0.0, 1.0, 1.0]
            })
            .draw(&pipeline, &vertex_buffer, &vulkano::command_buffer::DynamicState::none(),
                  &set, &())
            .draw_end()
            .build()
    }).collect::<Vec<_>>();

    'main: loop {
        let image_num = swapchain.acquire_next_image(Duration::new(10, 0)).unwrap();
        vulkano::command_buffer::submit(&command_buffers[image_num], &queue).unwrap();
        swapchain.present(&queue, image_num).unwrap();

        for ev in window.window().poll_events() {
            match ev {
                winit::Event::Closed => break 'main,
                _ => ()
            }
        }
    }

    // Free gl resources   
    gl::use_program(0);
    gl::disable_vertex_attrib_array(0);
    gl::disable_vertex_attrib_array(1);
    gl::disable_vertex_attrib_array(2);
    // gl::detach_shader(handles.program.get(), vertexshader);
    // gl::detach_shader(handles.program.get(), fragmentshader);
    // gl::delete_shader(vertexshader);
    // gl::delete_shader(fragmentshader);
    gl::delete_program(handles.program.get());
    gl::delete_buffers(&handles.vbos.get());
    gl::delete_vertex_arrays(&[handles.vao.get()]);

}
