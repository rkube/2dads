#version 120

attribute vec3 coord3d;
//uniform float widthx;
//uniform float widthy;
varying vec3 f_color;

void main(void)
{
    gl_Position = vec4(coord3d, 1.0);
    //gl_Position = vec4(coord3d.x / widthx, coord3d.y / widthy, coord3d.z, 1.0);
    f_color = coord3d;
}
