// Simplified ASCII background animation (no audio, no controls)

// Vertex shader (shared by both passes)
const vertexShaderSource = `#version 300 es
in vec2 a_position;
out vec2 v_uv;

void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

// Noise generation fragment shader
const noiseFragmentShaderSource = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_waveAmplitude;
uniform float u_noiseIntensity;
uniform float u_seed;
uniform vec2 u_resolution;

// Simplex 3D Noise
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  i = mod289(i);
  vec4 p = permute(permute(permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0));

  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);

  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;

  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

float fbm(vec3 p) {
  float value = 0.0;
  float amplitude = 1.0;
  float frequency = 1.0;

  for (int i = 0; i < 4; i++) {
    value += amplitude * snoise(p * frequency);
    frequency *= 2.0;
    amplitude *= 0.5;
  }

  return value;
}

void main() {
  vec2 uv = v_uv;
  float aspect = u_resolution.x / u_resolution.y;
  uv.x *= aspect;

  vec2 q = vec2(
    fbm(vec3(uv * 2.0 + u_seed, u_time * 0.3)),
    fbm(vec3(uv * 2.0 + u_seed + 100.0, u_time * 0.3))
  );

  vec2 r = vec2(
    fbm(vec3(uv * 2.0 + q * u_waveAmplitude + u_seed + 200.0, u_time * 0.2)),
    fbm(vec3(uv * 2.0 + q * u_waveAmplitude + u_seed + 300.0, u_time * 0.2))
  );

  vec2 warpedUV = uv + r * u_waveAmplitude;

  float noise = 0.0;
  noise += fbm(vec3(warpedUV * 1.0, u_time * 0.5)) * 1.0;
  noise += fbm(vec3(warpedUV * 2.0, u_time * 0.7)) * 0.5 * u_noiseIntensity;
  noise += fbm(vec3(warpedUV * 4.0, u_time * 0.9)) * 0.25 * u_noiseIntensity;

  noise = noise * 0.5 + 0.5;

  fragColor = vec4(vec3(noise), 1.0);
}
`;

// ASCII rendering fragment shader
const asciiFragmentShaderSource = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_noiseTex;
uniform float u_cellSize;
uniform float u_brightness;
uniform float u_contrast;
uniform float u_hue;
uniform float u_saturation;
uniform float u_vignette;
uniform float u_vignetteIntensity;
uniform float u_threshold1;
uniform float u_threshold2;
uniform float u_threshold3;
uniform float u_threshold4;
uniform float u_threshold5;
uniform vec2 u_resolution;

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float hexagon_sdf(vec2 p, float size) {
  vec3 k = vec3(-0.866025404, 0.5, 0.577350269);
  p = abs(p - 0.5);
  p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
  p -= vec2(clamp(p.x, -k.z * size, k.z * size), size);
  return step(length(p) * sign(p.y), 0.0);
}

void main() {
  vec2 pixelCoord = v_uv * u_resolution;

  // Hexagonal grid constants (flat-top hexagons)
  float hexRadius = u_cellSize * 0.5; // radius from center to vertex
  float hexWidth = hexRadius * 2.0; // vertex to vertex (horizontal)
  float hexHeight = hexRadius * 1.732050808; // sqrt(3) * radius (vertical flat-to-flat)

  // Horizontal and vertical spacing between hex centers
  float horizSpacing = hexWidth * 0.75; // 3/4 of width
  float vertSpacing = hexHeight;

  // Convert to hexagonal grid coordinates
  vec2 hexCoord = vec2(pixelCoord.x / horizSpacing, pixelCoord.y / vertSpacing);

  // Get approximate column and row
  float col = floor(hexCoord.x);
  float row = floor(hexCoord.y - mod(col, 2.0) * 0.5);

  // Calculate candidate hexagon centers (need to check neighboring cells)
  vec2 candidates[7];
  candidates[0] = vec2(col, row + mod(col, 2.0) * 0.5);
  candidates[1] = vec2(col + 1.0, row + mod(col + 1.0, 2.0) * 0.5);
  candidates[2] = vec2(col - 1.0, row + mod(col - 1.0, 2.0) * 0.5);
  candidates[3] = vec2(col, row + 1.0 + mod(col, 2.0) * 0.5);
  candidates[4] = vec2(col, row - 1.0 + mod(col, 2.0) * 0.5);
  candidates[5] = vec2(col + 1.0, row + 1.0 + mod(col + 1.0, 2.0) * 0.5);
  candidates[6] = vec2(col - 1.0, row + 1.0 + mod(col - 1.0, 2.0) * 0.5);

  // Find closest hexagon center
  vec2 closestHex = candidates[0];
  float minDist = 1e10;

  for (int i = 0; i < 7; i++) {
    vec2 hexCenterPixel = vec2(
      candidates[i].x * horizSpacing,
      candidates[i].y * vertSpacing
    );
    float dist = length(pixelCoord - hexCenterPixel);
    if (dist < minDist) {
      minDist = dist;
      closestHex = candidates[i];
    }
  }

  // Calculate pixel position of the closest hexagon center
  vec2 hexCenterPixel = vec2(closestHex.x * horizSpacing, closestHex.y * vertSpacing);

  // Calculate UV within the hexagon cell (normalized to 0-1)
  vec2 cellUV = (pixelCoord - hexCenterPixel) / hexWidth + 0.5;

  // Sample noise from hexagon center
  vec2 sampleUV = hexCenterPixel / u_resolution;
  float noise = texture(u_noiseTex, sampleUV).r;

  float brightness = noise * u_brightness;
  brightness = (brightness - 0.5) * u_contrast + 0.5;
  brightness = clamp(brightness, 0.0, 1.0);

  // Rotate cellUV by 60 degrees (π/3 radians = 1.047198)
  vec2 center = vec2(0.5);
  vec2 p = cellUV - center;
  float angle = 1.047198; // 60 degrees
  vec2 rotatedUV = vec2(
    p.x * cos(angle) - p.y * sin(angle),
    p.x * sin(angle) + p.y * cos(angle)
  ) + center;

  // Use brightness to determine hexagon size (0.0 to 0.45 range for smaller hexagons)
  float hexSize = brightness * 0.45;

  // Render the hexagon if it's above the first threshold
  float glyph = (brightness > u_threshold1) ? hexagon_sdf(rotatedUV, hexSize) : 0.0;

  vec3 color = hsv2rgb(vec3(u_hue / 360.0, u_saturation, brightness));

  float dist = length(v_uv - 0.5);
  float vignette = 1.0 - smoothstep(u_vignette - 0.3, u_vignette, dist) * u_vignetteIntensity;

  vec3 finalColor = color * glyph * vignette;

  fragColor = vec4(finalColor, 1.0);
}
`;

class WebGLRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = null;
    this.programs = {};
    this.framebuffer = null;
    this.noiseTexture = null;
    this.quadBuffer = null;
  }

  init() {
    this.gl = this.canvas.getContext('webgl2', {
      alpha: true,
      antialias: false,
      depth: false
    });

    if (!this.gl) {
      throw new Error('WebGL 2 is not supported');
    }

    const gl = this.gl;

    this.programs.noise = this.createProgram(vertexShaderSource, noiseFragmentShaderSource);
    this.programs.ascii = this.createProgram(vertexShaderSource, asciiFragmentShaderSource);

    this.createFramebuffer();
    this.createQuad();
    this.resize();

    window.addEventListener('resize', () => this.resize());
  }

  createShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error('Shader compilation error: ' + info);
    }

    return shader;
  }

  createProgram(vertexSource, fragmentSource) {
    const gl = this.gl;

    const vertexShader = this.createShader(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = this.createShader(gl.FRAGMENT_SHADER, fragmentSource);

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error('Program linking error: ' + info);
    }

    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);

    return program;
  }

  createFramebuffer() {
    const gl = this.gl;

    this.noiseTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.noiseTexture);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA,
      this.canvas.width, this.canvas.height, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, null
    );

    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D, this.noiseTexture, 0
    );

    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('Framebuffer is not complete');
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  createQuad() {
    const gl = this.gl;
    const vertices = new Float32Array([
      -1, -1, 1, -1, -1, 1,
      -1, 1, 1, -1, 1, 1
    ]);

    this.quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  }

  setUniforms(program, uniforms) {
    const gl = this.gl;

    for (const [name, value] of Object.entries(uniforms)) {
      const location = gl.getUniformLocation(program, name);
      if (location === null) continue;

      if (typeof value === 'number') {
        gl.uniform1f(location, value);
      } else if (Array.isArray(value)) {
        if (value.length === 2) {
          gl.uniform2fv(location, value);
        } else if (value.length === 3) {
          gl.uniform3fv(location, value);
        } else if (value.length === 4) {
          gl.uniform4fv(location, value);
        }
      }
    }
  }

  setupVertexAttribute(program, name) {
    const gl = this.gl;
    const location = gl.getAttribLocation(program, name);
    if (location === -1) return;

    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
  }

  render(state) {
    const gl = this.gl;

    // Pass 1: Render noise
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.programs.noise);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    this.setupVertexAttribute(this.programs.noise, 'a_position');

    this.setUniforms(this.programs.noise, {
      u_time: state.time,
      u_waveAmplitude: state.waveAmplitude,
      u_noiseIntensity: state.noiseIntensity,
      u_seed: state.seed,
      u_resolution: [state.width, state.height]
    });

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // Pass 2: Render ASCII
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.programs.ascii);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    this.setupVertexAttribute(this.programs.ascii, 'a_position');

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.noiseTexture);

    this.setUniforms(this.programs.ascii, {
      u_noiseTex: 0,
      u_cellSize: state.cellSize,
      u_brightness: state.brightness,
      u_contrast: state.contrast,
      u_hue: state.hue,
      u_saturation: state.saturation,
      u_vignette: state.vignette,
      u_vignetteIntensity: state.vignetteIntensity,
      u_threshold1: state.threshold1,
      u_threshold2: state.threshold2,
      u_threshold3: state.threshold3,
      u_threshold4: state.threshold4,
      u_threshold5: state.threshold5,
      u_resolution: [state.width, state.height]
    });

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  resize() {
    const gl = this.gl;
    const dpr = window.devicePixelRatio || 1;

    this.canvas.width = window.innerWidth * dpr;
    this.canvas.height = window.innerHeight * dpr;
    this.canvas.style.width = window.innerWidth + 'px';
    this.canvas.style.height = window.innerHeight + 'px';

    gl.bindTexture(gl.TEXTURE_2D, this.noiseTexture);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA,
      this.canvas.width, this.canvas.height, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, null
    );
    gl.bindTexture(gl.TEXTURE_2D, null);
  }
}

// Initialize and run
const canvas = document.getElementById('background-canvas');
const renderer = new WebGLRenderer(canvas);

const state = {
  // Static animation parameters (no audio)
  waveAmplitude: 0.3,
  noiseIntensity: 0.125,
  cellSize: 70,
  brightness: 0.8,
  contrast: 1.0,
  hue: 210,
  saturation: 0.3,
  vignette: 0.2,
  vignetteIntensity: 0.2,
  threshold1: 0.2,
  threshold2: 0.4,
  threshold3: 0.6,
  threshold4: 0.8,
  threshold5: 1.0,
  seed: Math.floor(Math.random() * 1000),
  time: 0,
  width: window.innerWidth,
  height: window.innerHeight
};

let lastTime = 0;

function animate(currentTime) {
  const deltaTime = lastTime === 0 ? 0 : (currentTime - lastTime) / 1000;
  lastTime = currentTime;

  // Update time with constant slow speed
  state.time += deltaTime * 0.2;

  // Update dimensions
  state.width = canvas.width;
  state.height = canvas.height;

  renderer.render(state);
  requestAnimationFrame(animate);
}

// Start animation
try {
  renderer.init();
  requestAnimationFrame(animate);
} catch (err) {
  console.error('Failed to initialize background animation:', err);
}
