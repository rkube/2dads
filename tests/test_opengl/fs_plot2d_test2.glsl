#version 120

varying vec3 f_color;
void main(void)
{
    // good for plotting theta
    //gl_FragColor = vec4(1.0 - 0.5 * f_color.z, 0.1, 0.5 * f_color.z, 0.0);
    // good for plotting strmf
    gl_FragColor = vec4(abs(f_color.z), 0.5, abs( f_color.z), 0.0);
}
