#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <GL/glew.h>
#include "shader.h"

using namespace std;


/// Load a shader from file, 
/// If everything is ok, reurn the shader_id
/// On error, return -1
GLuint load_shader_from_file(const char* shader_file_path, GLenum shaderType)
{
    // Create shader ID
    GLuint shader_id = glCreateShader(shaderType);
    // Source Code string of the shader
    string shader_code;

    // Result from compilation
    GLint compile_ok = GL_FALSE;
    // Info log from compilation
    int info_log_length;

    ifstream shader_stream(shader_file_path, std::ios::in);
    if(shader_stream.is_open())
    {
        string line = "";
        while(getline(shader_stream, line))
        {
            shader_code += "\n" + line;
        }
        shader_stream.close();
    }
    else
    {
        cerr << "Error loading shader code from " << shader_file_path << "\n\n";
        cerr << shader_code << "\n";
        return 666;
    }


    // Compile the shader code
    cout << "Compiling shader code from " << shader_file_path <<":\n";
    // pointer to raw string
    char const* shader_srcptr = shader_code.c_str();
    glShaderSource(shader_id, 1, &shader_srcptr, NULL);
    glCompileShader(shader_id);

    // Check Shader
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &compile_ok);
    glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &info_log_length);
    if (compile_ok ==  GL_FALSE)
    {
        vector<char> shader_errmsg(info_log_length + 1);
        glGetShaderInfoLog(shader_id, info_log_length, NULL, &shader_errmsg[0]);
        cout << &shader_errmsg[0] << "\n";
        return 666;
    }
    return shader_id;
}



GLuint LoadShaders(const char* vertex_file_path, const char* fragment_file_path)
{
    // Create shaders
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    // Read the Vertex Shader code from file
    string VertexShaderCode;
    ifstream VertexShaderStream(vertex_file_path, std::ios::in);
    if(VertexShaderStream.is_open())
    {
        string Line = "";
        while(getline(VertexShaderStream, Line))
            VertexShaderCode += "\n" + Line;
        VertexShaderStream.close();
    }
    else
    {
        cerr << "Error loading Vertex Shader code from " << vertex_file_path << "\n";
    }

    // Read the Fragment Shader code from file
    string FragmentShaderCode;
    ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
    if(FragmentShaderStream.is_open())
    {
        string Line = "";
        while(getline(FragmentShaderStream, Line))
            FragmentShaderCode += "\n" + Line;
        FragmentShaderStream.close();
    }
    else
    {
        cerr << "Error loading Fragment Shader code from " << fragment_file_path << "\n";
    }


    GLint Result = GL_FALSE;
    int InfoLogLength;

    // Compile Vertex Shader
    cout << "Compiling Vertex Shader: " << vertex_file_path << "\n";
    char const* VertexSourcePtr = VertexShaderCode.c_str();
    glShaderSource(VertexShaderID, 1, &VertexSourcePtr, NULL);
    glCompileShader(VertexShaderID);

    // Check Vertex Shader
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0)
    {
        vector<char> VertexShaderErrMsg(InfoLogLength+1);
        glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrMsg[0]);
        cout << &VertexShaderErrMsg[0] << "\n";
    }

    // Compile Fragment Shader
    cout << "Compiling Fragment Shader: " << fragment_file_path << "\n";
    char const* FragmentSourcePtr = FragmentShaderCode.c_str();
    glShaderSource(FragmentShaderID, 1, &FragmentSourcePtr, NULL);
    glCompileShader(FragmentShaderID);

    // Check Vertex Shader
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0)
    {
        vector<char> FragmentShaderErrMsg(InfoLogLength+1);
        glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrMsg[0]);
        cout << &FragmentShaderErrMsg[0] << "\n";
    }

    cout << "Linking program\n";
    GLuint ProgramID = glCreateProgram();
    glAttachShader(ProgramID, VertexShaderID);
    glAttachShader(ProgramID, FragmentShaderID);
    glLinkProgram(ProgramID);

    // Check the program
    glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
    glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if(InfoLogLength > 0)
    {
        vector<char> ProgramErrMsg(InfoLogLength+1);
        glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrMsg[0]);
        cout << &ProgramErrMsg[0] << "\n";
    }

    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);

    return ProgramID;
}

// End of file shader.cpp
