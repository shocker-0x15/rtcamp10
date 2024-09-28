#version 400
#extension GL_ARB_explicit_uniform_location : enable // required in version lower 4.3
#extension GL_ARB_shading_language_420pack : enable // required in version lower 4.2

layout(location = 0) uniform uvec2 imageSize;
layout(location = 1, binding = 0) uniform sampler2D srcTexture;

out vec4 colorF4;

void main(void) {
    vec2 srcPixel = gl_FragCoord.xy;
    srcPixel.y = imageSize.y - srcPixel.y;
    vec3 color = texelFetch(srcTexture, ivec2(srcPixel), 0).xyz;
    colorF4 = vec4(color, 1.0f);
}
