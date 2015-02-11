#version 120

attribute vec3 coord3d;
varying vec3 f_color;
uniform mat4 MVP;

void main(void)
{
    gl_Position = MVP * vec4(coord3d, 1.0);
    f_color = coord3d;
}
