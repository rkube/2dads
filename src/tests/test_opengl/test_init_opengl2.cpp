/*
 * Initialize a slab object, visualize the output with opengl
 *
 * Try to use opengl visualization
 *
 */


#include <iostream>
#include <vector>
#include <iomanip>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "slab_cuda.h"
#include "output.h"
#include "shader.h"

using namespace std;


class mode_struct{
    public:
        mode_struct(uint kx_, uint ky_) : kx(kx_), ky(ky_) {};
        uint kx;
        uint ky;
};


static void error_callback(int error, const char* description)
{
    cerr << description << "\n";
}


class mygl_context{
    public:
        mygl_context(int, int, slab_config&);
        void draw(float);
        void buffer_data(double*);
        void shutdown(int);
        ~mygl_context();

    private:
        GLFWwindow* window;
        GLuint program_id;
        GLint attr3d;
        GLuint vbo_buffer;
        GLuint vao;
        GLfloat* vertices;
        int Nx;
        int My;
        float x_left;
        float x_right;
        float delta_x;
        float y_lo;
        float y_up;
        float delta_y;
};


void mygl_context :: shutdown(int return_code)
{
    glfwTerminate();
}

mygl_context :: mygl_context(int window_width, int window_height, slab_config& my_config) :
    window(nullptr),
    program_id(0),
    attr3d(0),
    vbo_buffer(0),
    vao(0),
    vertices(nullptr),
    Nx(my_config.get_nx()),
    My(my_config.get_my()),
    x_left(my_config.get_xleft()),
    x_right(my_config.get_xright()),
    delta_x(my_config.get_deltax()),
    y_lo(my_config.get_ylow()),
    y_up(my_config.get_yup()),
    delta_y(my_config.get_deltay())
{

/* Initialize OpenGL context and creaet a window */
    GLint link_ok = GL_FALSE; // Linking error flag
    int info_log_length;
    GLuint vs_id;       // Vertex shader ID
    GLuint fs_id;       // Fragment shader ID

    cout << "Initialized:\n";
    cout << "Nx = " << Nx << "\tx_left = " << x_left << "\tx_right = " << x_right << "\tdelta_x = " << delta_x << "\n";
    cout << "My = " << My << "\ty_low = " << y_lo << "\ty_up = " << y_up << "\tdelta_y = " << delta_y << "\n";


    glfwSetErrorCallback(error_callback);
    if(glfwInit() != GL_TRUE)
    {
        cerr << "Unable to initialize glfw\n";
        shutdown(1);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    
    window = glfwCreateWindow (window_width, window_height, "Hello Triangle", NULL, NULL);
    if (!window)
    {
        cerr << "ERROR: could not open window with GLFW3\n";
        shutdown(1);
    }
    // Copy GLFWwindow to the context structure
    glfwMakeContextCurrent(window);
                                            
    // start GLEW extension handler
    glewExperimental = GL_TRUE;
    glewInit();
    // get version info
    cout << "Renderer: " << glGetString(GL_RENDERER) << "\n";
    cout << "OpenGL version supported: " << glGetString(GL_VERSION) << "\n";
  
    // tell GL to only draw onto a pixel if the shape is closer to the viewer
    glEnable(GL_DEPTH_TEST); // enable depth-testing
    glDepthFunc(GL_LESS); // depth-testing interprets a smaller value as "closer"

    program_id = glCreateProgram();   

    if ((vs_id = load_shader_from_file("vs_triangle.glsl", GL_VERTEX_SHADER)) == 666)
        shutdown(2);

    if ((fs_id = load_shader_from_file("fs_triangle.glsl", GL_FRAGMENT_SHADER)) == 666)
        shutdown(2);

    glAttachShader(program_id, vs_id);
    glAttachShader(program_id, fs_id);
    glLinkProgram(program_id);

    glGetProgramiv(program_id, GL_LINK_STATUS, &link_ok);
    glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &info_log_length);
    if(!link_ok)
    {
        vector<char> prog_err_msg(info_log_length + 1);
        glGetProgramInfoLog(program_id, info_log_length, NULL, &prog_err_msg[0]);
        cerr << "Error linking the program: " << &prog_err_msg[0] << "\n";
    }

    if ((attr3d = glGetAttribLocation(program_id, "coord3d")) == -1)
    {
        cerr << "Could not bind attribute: coord3d\n";
        shutdown(3);
    }

    vertices = new GLfloat[3 * Nx * My];
    //vertices = new GLfloat[9];

    GLfloat triangle_vertices[] = { 0.0f,  0.8f,  0.0f,
                                   -0.8f, -0.8f,  0.0f,
                                    0.8f, -0.8f,  0.0f};

    // Create a Vertex Buffer Object
    glGenBuffers(1, &vbo_buffer);
    // Tell OpenGL that the VBO is an Array Buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer);

    glBufferData(GL_ARRAY_BUFFER, sizeof(triangle_vertices), triangle_vertices, GL_STATIC_DRAW);
    glGenVertexArrays(1, &vao);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
}

void mygl_context :: buffer_data(cuda::real_t* buffer)
{
    //int m = 0;
    //int n = 0;
    //int idx = 0;
    //float x = 0.0;
    //float y = 0.0;
    //const int n_elem = 3 * Nx * My;
    //const float lengthx = x_right - x_left;
    //const float lengthy = y_up - y_lo;

    //// Create vertices
    //for(m = 0; m < My; m++)
    //{
    //    y = y_lo + ((float) m) * delta_y;   
    //    for(n = 0; n < Nx; n++)
    //    {
    //        x = x_left + ((float) n) * delta_x;
    //        idx = m * My + Nx;
    //        vertices[idx + 0] = x / lengthx;
    //        vertices[idx + 1] = y / lengthy;
    //        vertices[idx + 2] = 0.0f;
    //        //vertices[idx + 2] = ((GLfloat) buffer[idx]);
    //        cout << "Vertex (" << vertices[idx + 0] << ", " << vertices[idx + 1] << ", " << vertices[idx + 2] << ")\n";
    //    }
    //}
    // Buffer the data
    //glBufferData(GL_ARRAY_BUFFER, 9, vertices, GL_STATIC_DRAW);
    // Create Vertex Attribute Array
    //glBindVertexArray(vao);
    //glEnableVertexAttribArray(0);
    //glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer);
    //glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
}

void mygl_context :: draw(float c)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(program_id);   // Use the program, compiled and linked in the constructor
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glfwPollEvents();
    glfwSwapBuffers(window);

    cout << "end draw\n";
}


mygl_context :: ~mygl_context()
{
    delete [] vertices;
    glDeleteProgram(program_id);
    glDeleteBuffers(1, &vbo_buffer);
    glfwTerminate();
}




int main(void)
{
//    slab_config my_config;
//    my_config.consistency();


//    slab_cuda slab(my_config);
//    output_h5 slab_output(my_config);
//    // Create and populate list of modes specified in input.ini
//    vector<mode_struct> mode_list;
//    vector<double> initc = my_config.get_initc();
    
    //twodads::real_t time{0.0};
    //twodads::real_t delta_t{my_config.get_deltat()};

//    const unsigned int tout_full(ceil(my_config.get_tout() / my_config.get_deltat()));
    //const unsigned int num_tsteps(ceil(my_config.get_tend() / my_config.get_deltat()));
    //const unsigned int tlevs = my_config.get_tlevs();
    //unsigned int t = 1;

//    slab.init_dft();
//    slab.initialize();
//    //slab.rhs_fun(my_config.get_tlevs() - 1);
//    cout << "Output every " << tout_full << " steps\n";

    // Initialize OpenGL
    mygl_context context(640, 480, my_config);
    // Use a buffer to copy data from the CUDA device to host memory
//    cuda::real_t* buffer = new cuda::real_t[my_config.get_nx() * my_config.get_my()];
//    slab.get_data(twodads::field_t::f_theta, buffer);

    //context.buffer_data(buffer);
    // Print buffer contents. This should be the initialized data from the GPU
    //for(unsigned int m = 0; m < my_config.get_my(); m++)
    //{
    //    for(unsigned int n = 0; n < my_config.get_nx(); n++)
    //    {
    //        cout << buffer[m * my_config.get_nx() + n] << "\t";
    //    }
    //    cout << "\n";
    //}

    int foo;
    //context.draw(0.1);
    cout << "Enter a number: \n";
    cin >> foo;

    delete [] buffer;

    return(0);
}


    //slab_output.write_output(slab, time);

//    // Integrate the first two steps with a lower order scheme
//    for(t = 1; t < tlevs - 1; t++)
//    {
//        for(auto mode : mode_list)
//        {
//            cout << "mode with kx = " << mode.kx << " ky = " << mode.ky << "\n";
//            slab.integrate_stiff_debug(twodads::field_k_t::f_theta_hat, t + 1, mode.kx, mode.ky);
//        } 
//        slab.integrate_stiff(twodads::field_k_t::f_theta_hat, t + 1);
//
//        time += delta_t;
//        //slab.inv_laplace(twodads::field_k_t::f_omega_hat, twodads::field_k_t::f_strmf_hat, 0);
//        //slab.dft_c2r(twodads::field_k_t::f_theta_hat, twodads::field_t::f_theta, my_config.get_tlevs() - t - 1);
//        //slab.dft_c2r(twodads::field_k_t::f_strmf_hat, twodads::field_t::f_strmf, my_config.get_tlevs() - t);
//
//        //slab.write_output(time);
//        //slab.rhs_fun();
//        //slab.move_t(twodads::field_k_t::f_theta_rhs_hat, my_config.get_tlevs() - t - 1, 0);
//        //slab.move_t(twodads::field_k_t::f_omega_rhs_hat, my_config.get_tlevs() - t - 1, 0);
//        //outfile.str("");
//        //outfile << "theta_t" << setfill('0') << setw(5) << t << ".dat";
//        //slab.print_field(twodads::f_theta_hat, outfile.str());
//    }
//
//    for(; t < num_tsteps + 1; t++)
//    {
//        for(auto mode : mode_list)
//        {
//            cout << "mode with kx = " << mode.kx << " ky = " << mode.ky << "\n";
//            slab.integrate_stiff_debug(twodads::field_k_t::f_theta_hat, tlevs, mode.kx, mode.ky);
//        } 
//        slab.integrate_stiff(twodads::field_k_t::f_theta_hat, tlevs);
//        //slab.integrate_stiff(twodads::dyn_field_t::d_omega, tlevs);
//        //slab.dft_c2r(twodads::field_k_t::f_theta_hat, twodads::field_t::f_theta, 0);
//        //slab.dft_c2r(twodads::field_k_t::f_omega_hat, twodads::field_t::f_omega, 0);
//        slab.advance();
//        slab.update_real_fields(1);
//        time += delta_t;
//
//        if(t % tout_full == 0)
//            slab_output.write_output(slab, time);
//    }
//    return(0);
//}

