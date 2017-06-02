/*
 *
 * Example, taken from
 *
 * http://en.wikibooks.org/wiki/OpenGL_Programming
 *
 * Example 1: triangle
 *
 * Plot the surface from slab initialization as a texture
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

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
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
        GLuint vbo;             // Vertex buffer object
        GLuint vao_id;          // Vertex array object
        GLint attr_coord2d;     // Attribute corresponding to the coord3d variable in the vertex shader
        GLint uni_transf_text;  // Uniform variable corresponding to mat4 transform_text in the vs
        GLint uni_transf_vert;  // Uniform variable corresponding to mat4 transform_vert in the vs
        GLint uni_text;         // Uniform variable corresponding to sampler2d texture in the vs
        GLuint texture_id;      // Id of the texture we generate
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
        GLbyte* texture_data;   // Texture buffer
        GLfloat* vertex_data;   // Vertex buffer
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
    num_elements(2 * num_vertices),
    window(nullptr),
    texture_data(nullptr),
    vertex_data(nullptr)
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

    if ((vs_id = load_shader_from_file("vs_plot2d.glsl", GL_VERTEX_SHADER)) == 666)
    {
        cerr << "Unable to load vertex shader\n";
        shutdown(2);
    }

    if ((fs_id = load_shader_from_file("fs_plot2d.glsl", GL_FRAGMENT_SHADER)) == 666)
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

    if ((attr_coord2d = glGetAttribLocation(program_id, "coord2d")) == GL_INVALID_OPERATION)
    {
        cerr << "Could not bind attribute: coord2d\n";
        shutdown(3);
    }

    if ((uni_text = glGetUniformLocation(program_id, "texture")) == GL_INVALID_OPERATION)
    {
        cerr << "Could not bind uniform: texture\n";
        shutdown(3);
    }

    if ((uni_transf_text = glGetUniformLocation(program_id, "transform_text")) == GL_INVALID_OPERATION)
    {
        cerr << "Could not bind uniform: transform_text\n";
        shutdown(3);
    }

    if ((uni_transf_vert = glGetUniformLocation(program_id, "transform_vert")) == GL_INVALID_OPERATION)
    {
        cerr << "Could not bind uniform: transform_vert\n";
        shutdown(3);
    }

    cout << "attr_coord2d is " << attr_coord2d << "\n";
    cout << "texture is " << texture << "\n";
    cout << "transform_text is " << transform_text << "\n";
    cout << "transform_vert is " << transform_vert << "\n";
    cout << num_vertices << " vertices with a total of " << num_elements << " elements\n";

    vertex_data = new GLfloat[num_elements];
    texture_data = new GLbyte[num_vertices];
    for(m = 0; m < My; m++)
    {
        y = y_lo + ((float) m) * delta_y;
        for(n = 0; n < Nx; n++)
        {
            x = x_left + ((float) n) * delta_x;
            idx = 2 * (m * My + n);
            vertex_data[idx + 0] = x / (x_right - x_left);
            vertex_data[idx + 1] = y / (y_up - y_lo);
            texture_data[m * My + n] = roundf(0.0f);
        }
    }
    cout << "============================================================================\n";
    cout << "Vertex initialization: \n";
    for(n = 0; n < num_elements;n = n + 2)
        cout << "(" << vertex_data[n] << ", " << vertex_data[n + 1] << ")\n";
    cout << "============================================================================\n";
    
    // Upload the texture
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, N, N, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, texture_data);


    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, num_elements * sizeof(GLfloat), vertex_data, GL_STREAM_DRAW);

    // Define the attributes of the vertex buffer we just created
    // http://www.youtube.com/watch?v=7qSAVWJtcdI
    glGenVertexArrays(1, &vao_id);  // Declare one Vertex Array Object name, store it in vao_id
    glBindVertexArray(vao_id);      // Create and bind the Vertex Array Object, we just created
    glEnableVertexAttribArray(attr_coord3d);               // Enable the Vertex Attribute Array, stored in index 0
    glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer);             // Make the buffer which stores our triangles the current buffer
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); // Describe the data we put in the buffer to OpenGL
}

void mygl_context :: fill_buffer(double* data)
{
    int n{0};
    int m{0};
    int idx{0};
    for(m = 0; m < My; m++)
        for(n = 0; n < Nx; n++)
        {
            texture_data[m * My + n] = ((float) data[m * My + n]);
            //vertex_data[idx + 2] = 0.0f;
        }

    cout << "============================================================================\n";
    cout << "Vertex updates:\n";
    for(n = 0; n < num_elements; n = n + 2)
        cout << "(" << vertex_data[n] << ", " << vertex_data[n + 1] << ")\n";
    cout << "============================================================================\n";

    glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer);
    glBufferData(GL_ARRAY_BUFFER, num_elements * sizeof(GLfloat), vertex_data, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void mygl_context :: draw(float c)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    cout << "draw: vao_id = " << vao_id << "\n";
    glUseProgram(program_id);   // Use the program, compiled and linked in the constructor
    glUniform1i(uni_text, 0);

    // Create the MVP
    glm::mat4 model(1.0f);
    glm::mat4 view = glm::lookAt(glm::vec3(0.0, -2.0, 2.0), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 0.0, 1.0));
    glm::mat4 projection = glm::perspective(45.0f, 1.0f * 640 / 480, 0.1f, 10.0f);

    glm::mat4 vertex_transform = projection * view * model;
    glm::mat4 texture_transform = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(scale, scale, 1)), glm::vec3(0.0f, 0.0f, 0.0f));
    glUniformMatrix4fv(uni_transf_vert, 1, GL_FALSE, glm::value_ptr(transform_vert));
    glUniformMatrix4fv(uni_transf_text, 1, GL_FALSE, glm::value_ptr(transform_text));

     /* Set texture wrapping mode */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); 
    /* Set texture interpolation mode */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindVertexArray(vao_id);  // Bind the Vertex array. Setup of this vertex array object
                                // happens in the constructor
    glDrawArrays(GL_LINES, 0, 3);   // Draw triangles

    glfwPollEvents();
    glfwSwapBuffers(window);
    cout << "end draw\n";
}


mygl_context :: ~mygl_context()
{
    glDeleteProgram(program_id);
    glDeleteBuffers(1, &vbo_buffer);
    glfwTerminate();
    delete [] vertex_data;
    delete [] texture_data;
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


