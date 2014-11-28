/*
 *
 * Example, taken from
 *
 * http://en.wikibooks.org/wiki/OpenGL_Programming
 *
 * Example 1: triangle
 *
 * re-written as to use glfw
 *
 * See also: http://antongerdelan.net/opengl/hellotriangle.html
 */


#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "shader.h"
#include "slab_cuda.h"
#include "output.h"


using namespace std;

class mode_struct{
    public:
        mode_struct(uint kx_, uint ky_) : kx(kx_), ky(ky_) {};
        uint kx;
        uint ky;
};


class mygl_context{
    public:
        mygl_context(int, int, slab_config&);
        void draw(float);
        void fill_buffer(double*);
        void shutdown(int);
        ~mygl_context();
    private:

        GLuint program_id;      // program id
        GLuint vbo;             // Verex buffer object
        GLuint vao_id;          // Vertex array object
        GLint attr_coord3d;     // Attribute corresponding to the coord3d variable in the vertex shader
        //GLint uni_widthx;       // Uniform variable corresponding to widthx in the vertex shader
        //GLint uni_widthy;       // Uniform variable corresponding to widthy in the vertex shader
        const int Nx;
        const int My;
        const float x_left;
        const float x_right;
        const float delta_x;
        const float y_lo;
        const float y_up;
        const float delta_y;
        const int num_vertices; // Number of vertices
        const int num_elements; // Number of elements in the vertex buffer
        GLFWwindow* window;
        GLfloat* vertices;      // Vertex buffer
};


static void error_callback(int error, const char* description)
{
    cerr << description << "\n";
}

void mygl_context :: shutdown(int return_code)
{
    cerr << "Shutting down glfw\n";
    glfwTerminate();
}

mygl_context :: mygl_context(int window_width, int window_height, slab_config& my_config) :
    program_id(0),
    vbo(0),
    vao_id(0),
    attr_coord3d(0),
    Nx(my_config.get_nx()),
    My(my_config.get_my()),
    x_left(my_config.get_xleft()),
    x_right(my_config.get_xright()),
    delta_x(my_config.get_deltax()),
    y_lo(my_config.get_ylow()),
    y_up(my_config.get_yup()),
    delta_y(my_config.get_deltay()),
    num_vertices(Nx * My),
    num_elements(3 * num_vertices),
//    num_vertices(3),
//    num_elements(3 * num_vertices),
    window(nullptr),
    vertices(nullptr)
{
/* Initialize OpenGL context and creaet a window */
    GLint link_ok = GL_FALSE; // Linking error flag
    int info_log_length;
    GLuint vs_id;       // Vertex shader ID
    GLuint fs_id;       // Fragment shader ID

    int n{0};           // Loop variable for vertex initialization
    int m{0};           // Loop variable for vertex initialization
    int idx{0};         // Loop variable for vertex initialization
    float x{0.0f};      // Loop variable for vertex initialization
    float y{0.0f};      // Loop variable for vertex initialization

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
    //context.window = (GLFWwindow*) glfwGetWindowUserPointer(new_window);
    // Copy GLFWwindow to the context structure
    glfwMakeContextCurrent(window);
                                            
    // start GLEW extension handler
    glewExperimental = GL_TRUE;
    glewInit();
    // get version info
    cout << "Renderer: " << glGetString(GL_RENDERER) << "\n";
    cout << "OpenGL version supported: " << glGetString(GL_VERSION) << "\n";
  
    // tell GL to only draw onto a pixel if the shape is closer to the viewer
    glEnable (GL_DEPTH_TEST); // enable depth-testing
    glDepthFunc (GL_LESS); // depth-testing interprets a smaller value as "closer"

    program_id = glCreateProgram();   

    if ((vs_id = load_shader_from_file("vs_triangle.glsl", GL_VERTEX_SHADER)) == 666)
    {
        cerr << "Unable to load vertex shader\n";
        shutdown(2);
    }

    if ((fs_id = load_shader_from_file("fs_triangle.glsl", GL_FRAGMENT_SHADER)) == 666)
    {
        cerr << "Unable to load fragment shader\n";
        shutdown(2);
    }

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

    if ((attr_coord3d = glGetAttribLocation(program_id, "coord3d")) == GL_INVALID_OPERATION)
    {
        cerr << "Could not bind attribute: coord3d\n";
        shutdown(3);
    }

    //if ((uni_widthx = glGetUniformLocation(program_id, "widthx")) == GL_INVALID_OPERATION)
    //{
    //    cerr << "Could not bind uniform: widthx\n";
    //    shutdown(3);
    //}

    //if ((uni_widthy = glGetUniformLocation(program_id, "widthy")) == GL_INVALID_OPERATION)
    //{
    //    cerr << "Could not bind uniform: widthy\n";
    //    shutdown(3);
    //}

    cout << "attr_coord3d is " << attr_coord3d << "\n";
    //cout << "uni_widthx is " << uni_widthx << "\n";
    //cout << "uni_widthy is " << uni_widthy << "\n";
    cout << num_vertices << " vertices with a total of " << num_elements << " elements\n";

    vertices = new GLfloat[num_elements];
    //for(int i = 0; i < num_elements; i++)
    //    vertices[i] = 0.0f;
    for(m = 0; m < My; m++)
    {
        y = y_lo + ((float) m) * delta_y;
        for(n = 0; n < Nx; n++)
        {
            x = x_left + ((float) n) * delta_x;
            idx = 3 * (m * My + n);
            vertices[idx + 0] = x / (x_right - x_left);
            vertices[idx + 1] = y / (y_up - y_lo);
            vertices[idx + 2] = 0.0f;
        }
    }
    //cout << "============================================================================\n";
    //cout << "Vertex initialization: \n";
    //for(n = 0; n < num_elements;n = n + 3)
    //    cout << "(" << vertices[n] << ", " << vertices[n + 1] << ", " << vertices[n + 2] << ")\n";
    //cout << "============================================================================\n";


    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, num_elements * sizeof(GLfloat), vertices, GL_STREAM_DRAW);

    // Define the attributes of the vertex buffer we just created
    // http://www.youtube.com/watch?v=7qSAVWJtcdI
    glGenVertexArrays(1, &vao_id);  // Declare one Vertex Array Object name, store it in vao_id
    glBindVertexArray(vao_id);      // Create and bind the Vertex Array Object, we just created
    glEnableVertexAttribArray(attr_coord3d);               // Enable the Vertex Attribute Array, stored in index 0
    glBindBuffer(GL_ARRAY_BUFFER, vbo);                    // Make the buffer which stores our triangles the current buffer
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); // Describe the data we put in the buffer to OpenGL
    //glUniform1f(uni_widthx, 0.8f);
    //glUniform1f(uni_widthy, 0.8f);
    //glUniform1f(uni_widthx, x_right - x_left);
    //glUniform1f(uni_widthy, y_up - y_lo);
}

void mygl_context :: fill_buffer(double* data)
{
    int n{0};
    int m{0};
    int idx{0};
    for(m = 0; m < My; m++)
        for(n = 0; n < Nx; n++)
        {
            idx = 3 * (m * My + n);
            vertices[idx + 2] = ((float) data[m * My + n]);
            //vertices[idx + 2] = 0.0f;
        }

    //cout << "============================================================================\n";
    //cout << "Vertex updates:\n";
    //for(n = 0; n < num_elements; n = n + 3)
    //    cout << "(" << vertices[n] << ", " << vertices[n + 1] << ", " << vertices[n + 2] << ")\n";
    //cout << "============================================================================\n";

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, num_elements * sizeof(GLfloat), vertices, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void mygl_context :: draw(float c)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    cout << "draw: vao_id = " << vao_id << "\n";
    glUseProgram(program_id);   // Use the program, compiled and linked in the constructor
    glBindVertexArray(vao_id);  // Bind the Vertex array. Setup of this vertex array object
                                // happens in the constructor
    //glUniform1f(uni_widthx, 1.0f);
    //glUniform1f(uni_widthy, 1.0f);
    glDrawArrays(GL_POINTS, 0, num_vertices);   // Draw triangles

    glfwPollEvents();
    glfwSwapBuffers(window);
    cout << "end draw\n";
}


mygl_context :: ~mygl_context()
{
    glDeleteProgram(program_id);
    glDeleteBuffers(1, &vbo);
    glfwTerminate();
    delete [] vertices;
}


void mygl_free_resources()
{
}

int main(void)
{
    int foo;
    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);
    output_h5 slab_output(my_config);
    // Create and populate list of modes specified in input.ini
    vector<mode_struct> mode_list;
    vector<double> initc = my_config.get_initc();
    
    //twodads::real_t time{0.0};
    //twodads::real_t delta_t{my_config.get_deltat()};

    const unsigned int tout_full(ceil(my_config.get_tout() / my_config.get_deltat()));
    //const unsigned int num_tsteps(ceil(my_config.get_tend() / my_config.get_deltat()));
    //const unsigned int tlevs = my_config.get_tlevs();
    //unsigned int t = 1;

    slab.init_dft();
    slab.initialize();
    //slab.rhs_fun(my_config.get_tlevs() - 1);
    cout << "Output every " << tout_full << " steps\n";

    // Initialize OpenGL
    // Use a buffer to copy data from the CUDA device to host memory
    cuda::real_t* buffer = new cuda::real_t[my_config.get_nx() * my_config.get_my()];
    slab.get_data(twodads::field_t::f_theta, buffer);
    mygl_context context(640, 480, my_config);

    //double data_buffer[9] = {
    //     0.0,  0.8, 0.0,
    //    -0.8, -0.8, 0.0,
    //     0.8, -0.8, 0.0};

    context.draw(0.1);

    for(int i = 0; i < 5; i++)
    {
        cout << "Enter a number: \n";
        cin >> foo;
        //for(int j = 0; j < 9; j++)
        //    data_buffer[j] *= 0.9;

        cin.ignore(numeric_limits<std::streamsize>::max(), '\n');
        context.fill_buffer(buffer);
        context.draw(0.1 * i);
    }
    return(0);
}


