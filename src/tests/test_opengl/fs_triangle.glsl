#version 120

varying vec3 f_color;
void main(void)
{
    //gl_FragColor = vec4(0.2, 0.3, 0.4, 0.0);
    gl_FragColor = vec4(f_color.z, 0.3, 0.3, 0.0);
}
